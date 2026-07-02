"""
运行并评估模型生成的代码输出。大致流程如下：

1. 读取输入（原始较慢的代码）和模型生成的程序，并将它们写入临时目录。
2. 多次运行每个程序，并计算平均耗时和准确率。
3. 将结果写入 json 文件中进行报告输出。

示例用法（通过环境变量传入参数并执行脚本）：
export REF_FILE="data/codenet/splits/problem_id/2023-01-13_12-56pm/seq2seq_splits/test.jsonl" && export OP="/usr1/amadaan/learning2perf/data/outputs/beam_outputs_codegen_2b_jan_13_split.jsonl" && export CPU=30 && nohup python -u src/codenet_eval/run_eval.py  --model_generated_outputs_path ${OP} --reference_file_path ${REF_FILE} --output_report_file_path ${OP}.25_tries.report --slow_code_col input --model_generated_potentially_faster_code_col beam_generated_target_from_input --reference_code_col target --num_problems_to_evaluate -1 --cpu_number $CPU --max_time_per_run 10 --num_trials 25 --ignore_first_k 2

"""




"""

这份代码是一个非常完整且严谨的自动化代码评测脚本。它的核心目的是：评估大语言模型（LLM）生成的代码是否比原始代码运行得更快，且保证逻辑正确。

简单来说，它就像一个“代码性能裁判”，让三名选手（原始慢代码、模型优化后的代码、标准答案代码）在同一个标准测试环境里跑题，最后给出一份详尽的成绩单。

以下是该代码功能的详细结构分析：

1. 核心工作流程 (Core Workflow)
整个脚本的运行逻辑可以提炼为以下 5 个步骤：

数据合并与清洗：读取原始数据集（包含较慢的代码和测试用例）以及 LLM 生成的结果集，将它们对齐合并。如果模型对同一道题生成了多个版本的代码（例如使用了 Beam Search），脚本也会将其展开准备测试。

构建物理测试环境：为了避免代码在运行时由于重复读取系统 I/O 而产生的时间波动，脚本会先在内存/临时目录中将所有待测代码写成真实的 .py 或 .cpp 文件，并把正确答案（Ground Truth）预先加载到内存中。

沙盒执行与计时：这是最核心的一步。调用外部沙盒模块 (sandbox_o) 运行代码。为了保证测试的科学性，它会：

运行多次取平均值 (num_trials)。

忽略前几次运行的结果 (ignore_first_k)，以排除程序冷启动、缓存预热等带来的时间偏差。

多解寻优（针对生成模型）：如果模型生成了 5 个不同的优化方案，脚本会找出完全通过所有测试用例（准确率 100%）且运行时间最短的那一个作为最终成绩。

多维数据分析与报告：生成一份详细的 JSONL 报告，并在控制台打印关键统计指标，比如多少个代码被成功优化、各项代码的平均耗时等。

2. 关键函数解析 (Key Functions)
evaluate_generated_outputs(): 主控函数，串联所有流程。

read_inputs_and_prepare(): 数据准备。它还有一个隐藏的高级功能——断点续传（通过 _prepare_for_rerun）。如果程序跑到一半崩溃了，再次传入 .report 文件时，它会过滤掉已经测过的高分代码，只测剩下的。

write_programs_read_ground_truth(): I/O 管理。创建临时文件夹（tempfile.TemporaryDirectory），把三类代码存进去，这一步是为了给后续的沙盒提供干净、可执行的物理路径。

run_programs(): 测试引擎。它遍历每一题，把沙盒环境跑出的 avg_time（平均时间）、std_time（时间方差）和 avg_acc（准确率）记录下来。如果有超时或报错（抛出 Exception），会自动记为 0 分或空值，防止程序中断。

print_summary(): 裁判宣判。它过滤掉准确率不达标（< 0.99）的废弃数据，重点对比 input_time_mean (原代码耗时) 和 gen_time_mean (生成代码耗时)。

3. 代码的工程亮点
除了基础的评测功能外，这段代码在工程实践上有几个非常专业的设计：

严谨的基准测试 (Benchmarking) 策略：

没有直接用 Python 的 time.time() 简单测试，而是委托给沙盒，并设置了 ignore_first_k（预热）和 num_trials（多次求均值）。这在底层性能测试（尤其是 C++ 和 Python）中是消除系统抖动（Jitter）的标准操作。

异常检测机制 (get_anomalies)：

数据集（如 CodeNet）中宣称某个参考代码比原代码快，但在作者当前的沙盒机器上实测时，参考代码反而更慢。代码通过 measured_rel_improvement < 0 识别出了这些“环境不一致”或“数据集标注有误”的脏数据。

安全性隔离：

使用了 sandbox_o.run_code_on_inputs，这意味着它假设模型生成的代码可能是恶意的（如包含 rm -rf / 或死循环），因此限制了最高 CPU 数（cpu_number）、最长运行时间（max_time_per_run），并在沙盒中安全执行。

宽容度设计 (*1.1 阈值)：

在最后统计“优化成功的样本数”时，判断条件是 gen_time * 1.1 < slow_time。这意味着生成的代码必须要比原代码快至少 10% 才算真正的优化成功，排除了极小的时间波动造成的误判。

"""
import pathlib
import tempfile
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple
import os
import logging
import glob
import numpy as np
from collections import defaultdict

# 导入自定义配置类和沙盒运行模块
from evalconfig import EvaluationConfig
from sandbox_o import run_code_on_inputs

import pdb

# 设置日志级别为 CRITICAL，只打印最严重的错误，避免输出过多调试信息
logging.basicConfig(level=logging.CRITICAL)

# 语言到文件后缀的映射字典
lang2file_ending = {
    "python": "py", 
    "cpp": "cpp"
}
    
def evaluate_generated_outputs(cfg: EvaluationConfig) -> None:
    """评估模型生成的程序的准确率和运行时间。有关详细信息，请参阅 EvaluationConfig 类的文档。"""

    # 第 0 步：读取和准备数据
    # 将输入数据（参考的较慢代码）与模型生成的代码合并为一个 DataFrame
    merged = read_inputs_and_prepare(cfg)
    print(f"待评估的程序数量: {len(merged)}")
    print(f"每个程序的运行次数 (num_trials): {cfg.num_trials}")
    print(f"忽略的前 K 次运行 (用于预热): {cfg.ignore_first_k}")
    print(f"每次运行的最大时间限制: {cfg.max_time_per_run}")
    print(f"输入列名 (较慢的代码): {cfg.slow_code_col}")
    print(f"参考列名 (已知较快的代码): {cfg.reference_code_col}")
    print(f"模型生成的列名 (可能更快的代码): {cfg.model_generated_potentially_faster_code_col}")
    print(f"输入/输出测试用例的基础路径: {cfg.inputs_outputs_basepath}")

    # 第 1 步：将输入代码和模型生成的代码以文件的形式写入临时目录，并读取测试用例 (Ground Truth)
    problem_id_to_ground_truths, output_code_location = write_programs_read_ground_truth(
        cfg, merged
    )

    # 第 2 步：运行程序
    
    # 确定文件后缀（如 .py 或 .cpp）
    lang_file_ending = lang2file_ending[cfg.language]
    
    # 定义需要运行的代码类型及其对应的文件后缀标签
    # tag_to_path 是一个列表，包含 (DataFrame 中的列名/标签, 文件后缀)
    tag_to_path = [
        ("input", f"_slow.{lang_file_ending}"),            # 原始较慢代码
        ("reference", f"_reference.{lang_file_ending}"),  # 参考较快代码
    ]
    
    # 检查模型对同一个输入是否生成了多个输出 (例如使用了束搜索 Beam Search 产生多个结果)
    is_multigen = isinstance(merged[cfg.model_generated_potentially_faster_code_col].iloc[0], list)
    if is_multigen:
        # 如果是多生成结果，将每个生成结果都加入待运行的列表
        num_generations = len(merged[cfg.model_generated_potentially_faster_code_col].iloc[0])
        tag_to_path.extend([(f"{cfg.model_generated_potentially_faster_code_col}_{i}", f"_maybe_faster_{i}.{lang_file_ending}") for i in range(num_generations)])
    else:
        # 如果只有一个生成结果
        tag_to_path.append((cfg.model_generated_potentially_faster_code_col, f"_maybe_faster_0.{lang_file_ending}"))
    
    # 在沙盒中实际运行这些程序，并返回执行结果（耗时、准确率等）
    results = run_programs(cfg, merged, problem_id_to_ground_truths, output_code_location, tag_to_path)
    
    # 如果是多生成结果，需要筛选出表现最好的那一个（准确率为1.0且耗时最短）
    if is_multigen:
        results = get_best_generation_per_submission(results, gen_col=cfg.model_generated_potentially_faster_code_col)
    
    # 第 3 步：总结运行结果，并写入报告
    print_summary(cfg, merged, results, gen_col=cfg.model_generated_potentially_faster_code_col)

    # 如果使用了临时目录，运行结束后进行清理
    if isinstance(cfg.temp_dir, tempfile.TemporaryDirectory):
        cfg.temp_dir.cleanup()


def read_inputs_and_prepare(cfg) -> pd.DataFrame:
    """读取模型生成的输出和参考文件，将它们连接 (join) 起来，并返回包含合并数据的 DataFrame。"""
    print(f"从 {cfg.reference_file_path} 读取参考文件")
    print(f"从 {cfg.model_generated_outputs_path} 读取模型生成的输出")

    print(
        f"每个程序运行 {cfg.num_trials} 次，跳过前 {cfg.ignore_first_k} 次运行，并从 {cfg.inputs_outputs_basepath} 获取输入输出对"
    )
    
    # 读取模型生成的数据 (JSON Lines 格式)
    gen_df = pd.read_json(
        cfg.model_generated_outputs_path, lines=True, orient="records"
    )
    # 恢复中断的执行：如果传入的是 .report 结尾的文件，说明是重新运行，直接过滤出需要继续跑的数据
    if cfg.model_generated_outputs_path.endswith(".report"):
        return _prepare_for_rerun(gen_df, cfg)
    
    print(f"从生成结果文件读取了 {len(gen_df)} 行")
    
    # 提取“较慢的代码”。如果使用的是基于 prompt 的输入，需要从 prompt 中解析出代码
    if cfg.is_prompt_based:
        gen_df["slower_program"] = gen_df.apply(
            lambda x: get_input_from_prompt(x), axis=1
        )
    else:
        # 否则直接从指定的列读取代码并去除首尾空白
        gen_df["slower_program"] = gen_df[cfg.slow_code_col].apply(lambda x: x.strip())
        

    if cfg.reference_file_path is not None:
        # 读取参考（Ground Truth）文件
        ref_df = pd.read_json(cfg.reference_file_path, lines=True, orient="records")
        ref_df["slower_program"] = ref_df["input"].apply(
            lambda x: x.strip().replace("\n\n\n\n\n", "")
            # TODO: 移除这个硬编码的 hack (处理 prompt-lib 未保留完整输入的问题)
        )
        
        print(f"参考文件中唯一的输入数量: {len(ref_df['slower_program'].unique())}")
        gen_df["slower_program"] = gen_df[
            "slower_program"
        ].apply(lambda x: x.strip().replace("\n\n\n\n\n", ""))
        
        # 确保每个 submission_id 是唯一的
        assert len(ref_df["submission_id_v0"].unique()) == len(
            ref_df
        ), "submission_id_v0 必须是唯一的"

        # 根据原始代码 (slower_program) 将生成结果和参考结果做内连接 (Inner Join)
        merged = pd.merge(
            gen_df,
            ref_df,
            left_on="slower_program",
            right_on="slower_program",
            suffixes=("", "_ref"),
            how="inner",
        )

        # 去重，保证每个“慢代码”只评估一次
        merged = merged.drop_duplicates(subset=["slower_program"])
        
        # 检查合并是否丢失了过多数据，防止输入代码在两边不一致导致的 bug
        assert abs(len(merged) - len(gen_df)) < 10, f"合并操作不应丢失过多行！请检查输入是否一致。合并丢失了 {len(gen_df) - len(merged)} 行。 len(gen_df)={len(gen_df)}, len(merged)={len(merged)}"
    else:
        # 如果没有提供外部参考文件，则假定生成结果文件本身已经包含了参考代码
        assert (
            cfg.reference_code_col in gen_df.columns
        ), f"在 {cfg.model_generated_outputs_path} 中找不到列 {cfg.reference_code_col}"
        merged = gen_df
        
        # 过滤掉较慢代码和参考代码完全相同的行（没有优化的意义）
        merged = merged[merged[cfg.slow_code_col] != merged[cfg.reference_code_col]]

    assert (
        len(merged) > 0
    ), f"所有程序中，{cfg.slow_code_col} 和 {cfg.reference_code_col} 都是相同的，无数据可评估。"
    
    # 限制评估的数量，用于快速调试
    if cfg.num_problems_to_evaluate != -1:
        merged = merged[: cfg.num_problems_to_evaluate]
    
    
    # 如果生成的代码是一个列表（多生成场景），为每个生成的代码添加一个独立的列
    if isinstance(merged[cfg.model_generated_potentially_faster_code_col].iloc[0], list):
        num_generations = len(merged[cfg.model_generated_potentially_faster_code_col].iloc[0])
        for i in range(num_generations):
            merged[f"{cfg.model_generated_potentially_faster_code_col}_{i}"] = merged[cfg.model_generated_potentially_faster_code_col].apply(lambda x: x[i])
    return merged


def _prepare_for_rerun(df: pd.DataFrame, cfg: EvaluationConfig) -> pd.DataFrame:
    """处理断点续传的逻辑，过滤掉已经成功运行（准确率>0.99）的数据"""
    acc_columns = {"generated_answers_acc", "generated_answer_acc"}
    # 找到存在的准确率列
    acc_column = list(acc_columns.intersection(set(df.columns)))[0]
    print("警告！这是一次重新运行 (RERUN)")
    print("正在准备重新运行...")
    print(f"找到准确率列: {acc_column}, 共 {len(df)} 行")
    
    # 选取之前准确率很高的行（这里逻辑可能是需要重新跑通过验证的，或者作者为了过滤脏数据）
    df = df[df[acc_column] > 0.99]
    # 移除之前计算出的 mean, std, acc 结果列，确保重新计算时不冲突
    df = df[[c for c in df.columns if not any(x in c for x in ["mean", "std", "acc"])]]
    print(f"过滤后剩余 {len(df)} 行")
    if cfg.num_problems_to_evaluate != -1:
        df = df[: cfg.num_problems_to_evaluate]
    return df
    

def write_programs_read_ground_truth(
    cfg: EvaluationConfig, merged: pd.DataFrame
) -> Tuple[Dict[str, List[str]], str]:
    """将所有待测试的程序写入临时目录，并读取测试用例 (ground truth)。
    为了减少运行时的 I/O 导致的时间方差，我们需要提前将测试程序的代码写入文件。
    返回：测试用例字典，和代码存放的目录路径。
    """
    problem_id_to_ground_truths = defaultdict(list)
    
    # 创建临时目录或使用指定的目录
    if cfg.temp_dir is None:
        cfg.temp_dir = tempfile.TemporaryDirectory()
        output_code_location = cfg.temp_dir.name
    else:
        output_code_location = cfg.temp_dir
        pathlib.Path(output_code_location).mkdir(parents=True, exist_ok=True)

    # 遍历合并后的数据帧
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="writing programs"):
        problem_id = row["problem_id"]

        # 1. 读取该题目的预期输出 (Ground Truth)
        if problem_id not in problem_id_to_ground_truths:
            num_test_cases = len(
                glob.glob(f"{cfg.inputs_outputs_basepath}/{problem_id}/output*.txt")
            )
            assert (
                num_test_cases > 0
            ), f"{cfg.inputs_outputs_basepath}/{problem_id} 目录下没有 ground truth 文件！"
            
            # 将该题目所有的测试用例输出读入字典
            for i in range(num_test_cases):
                with open(f"{cfg.inputs_outputs_basepath}/{problem_id}/output.{i}.txt") as f:
                    problem_id_to_ground_truths[problem_id].append(f.read().strip() + "\n")

        # 2. 将生成代码、原代码、参考代码写入临时目录
        lang_file_ending = lang2file_ending[cfg.language]
        submission_id_v0 = row["submission_id_v0"]
        
        # 写入原程序 (较慢的)
        with open(
            os.path.join(output_code_location, f"{submission_id_v0}_{problem_id}_slow.{lang_file_ending}"), "w"
        ) as f:
            f.write(row["slower_program"])

        # 处理生成程序有多个的情况，并依次写入文件
        generated_programs = row[cfg.model_generated_potentially_faster_code_col]
        if isinstance(generated_programs, str):
            generated_programs = [generated_programs]
        
        for i, generated_program in enumerate(generated_programs):
            with open(
                os.path.join(output_code_location, f"{submission_id_v0}_{problem_id}_maybe_faster_{i}.{lang_file_ending}"),
                "w"
            ) as f:
                f.write(generated_program.strip())

        # 写入参考程序 (较快的)
        with open(
            os.path.join(output_code_location, f"{submission_id_v0}_{problem_id}_reference.{lang_file_ending}"), "w"
        ) as f:
            f.write(row[cfg.reference_code_col].strip())

    print(f"所有程序已成功写入到 {output_code_location}")
    return problem_id_to_ground_truths, output_code_location


def run_programs(
    cfg: EvaluationConfig,
    merged: pd.DataFrame,
    problem_id_to_ground_truths: Dict,
    output_code_location: str,
    tag_to_path
):
    """实际执行程序并记录运行指标。

    参数:
        cfg (EvaluationConfig): 评估配置参数。
        merged (pd.DataFrame): 合并后的数据帧。
        problem_id_to_ground_truths (Dict): 每道题的预期输出结果。
        output_code_location (str): 代码存放的目录。
        tag_to_path: 标签和对应文件后缀的元组列表。

    返回:
        Dict: 包含每个提交 (submission) 的执行状态字典。
    """

    results = dict()
    # 确保每个行都有唯一的 submission_id_v0 用作结果字典的键
    assert len(merged["submission_id_v0"].unique()) == len(
        merged
    ), f"每一行必须有唯一的 submission_id_v0。当前不满足：独立数量 {len(merged['submission_id_v0'].unique())}, 总行数 {len(merged)}"
    
    # 逐条运行程序
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="running programs"):
        problem_id = row["problem_id"]
        submission_id_v0 = row["submission_id_v0"]
        unit_test_data_basepath = f"{cfg.inputs_outputs_basepath}/{problem_id}"
        
        try:
            problem_execution_stats = dict()
            # 运行生成的程序、原程序(slower)和参考程序(reference)
            for (tag, suffix) in tag_to_path:
                code_path = os.path.join(
                    output_code_location, f"{submission_id_v0}_{problem_id}{suffix}"
                )
                
                # 调用沙盒安全地运行代码并获得平均用时、用时标准差和通过的准确率
                avg_time, std_time, avg_acc = run_code_on_inputs(  # type: ignore
                    language=cfg.language,
                    code_path=code_path,
                    ground_truths=problem_id_to_ground_truths[problem_id],
                    unit_test_data_basepath=unit_test_data_basepath,
                    num_runs_per_test_case=cfg.num_trials,
                    ignore_first_k=cfg.ignore_first_k,
                    max_seconds_per_run=cfg.max_time_per_run,
                    cpu_number=cfg.cpu_number,
                    cflags=cfg.cflags,
                    return_if_acc_below=cfg.return_if_acc_below,
                )

                # 记录运行指标
                problem_execution_stats.update(
                    {
                        f"{tag}_time_mean": avg_time,
                        f"{tag}_time_std": std_time,
                        f"{tag}_acc": avg_acc,
                    }
                )
            results[submission_id_v0] = problem_execution_stats

        except Exception as e:
            # 捕获沙盒或执行过程中抛出的异常，将成绩记录为空或0
            logging.error(e)
            tmp = dict()
            for tag, suffix in tag_to_path:
                tmp[f"{tag}_time_mean"] = np.nan
                tmp[f"{tag}_time_std"] = np.nan
                tmp[f"{tag}_acc"] = 0.0
            results[submission_id_v0] = tmp
            continue

    print(f"成功运行了 {len(results)} 个问题")
    return results


def get_best_generation_per_submission(results: Dict, gen_col: str):
    """如果有多个生成的代码（例如通过Beam Search），获取每个提交中表现最好的一个。
    这里的“最好”被定义为在代码完全正确 (acc == 1.0) 的前提下，运行时间最短的那个。

    参数:
        results (Dict): 运行程序后的结果字典。
        gen_col (str): 模型生成代码的基础列名。

    返回:
        Dict: 更新后只保留了最佳生成代码指标的 results 字典。
    """
    best_per_sub = dict()
    for submission_id_v0, result_dict in results.items():
        # 找出所有包含模型生成列并且包含平均时间的键值对
        gen_op_times = [(k, v) for k, v in result_dict.items() if gen_col in k and "time_mean" in k]
        # 按运行时间从小到大排序
        gen_op_times = sorted(gen_op_times, key=lambda x: x[1])
        
        # 遍历排序好的列表，找到第一个准确率为 1.0（即测试用例全过）的代码
        for gen_op_time in gen_op_times:
            # 构造对应的 acc 键名
            if result_dict[f"{gen_op_time[0].replace('_time_mean', '')}_acc"] == 1.0:
                gen_op_times = [gen_op_time] # 找到即保留这一个并跳出
                break
                
        # 将最好的这一组数据提取出来，并保存回通用命名的字段中（不带 _0, _1 这样的后缀）
        try: 
            best_gen_key = gen_op_times[0][0].replace("_time_mean", "")
            best_per_sub[submission_id_v0] = result_dict
            best_per_sub[submission_id_v0][f"{gen_col}_time_mean"] = gen_op_times[0][1]
            best_per_sub[submission_id_v0][f"{gen_col}_time_std"] = result_dict[f"{best_gen_key}_time_std"]
            best_per_sub[submission_id_v0][f"{gen_col}_acc"] = result_dict[f"{best_gen_key}_acc"]
        except IndexError:
            # 如果所有生成代码都超时或报错导致没有时间数据，进入 pdb 调试模式
            pdb.set_trace()

    return best_per_sub

def print_summary(cfg, merged, results, gen_col: str):
    """打印执行结果的总结指标，并保存详细报告到文件。"""
    report_rows = []
    
    # 将原始的 DataFrame 数据和刚刚评估得到的 results 字典合并在一起
    for _, row in tqdm(merged.iterrows(), total=len(merged)):
        submission_id_v0 = row["submission_id_v0"]

        if submission_id_v0 not in results:
            continue

        report_row = row.to_dict()
        report_row.update(results[submission_id_v0])
        report_rows.append(report_row)

    assert len(results) == len(report_rows)
    print(f"正在将包含 {len(report_rows)} 行的报告写入到 {cfg.output_report_file_path}")
    run_metrics = pd.DataFrame(report_rows)
    
    # 将数据以 JSON Lines 的格式保存到文件
    run_metrics.to_json(cfg.output_report_file_path, orient="records", lines=True)

    # 过滤数据进行最终的统计：只考虑生成代码和原慢代码均通过测试 (>0.99) 的情况
    run_metrics = run_metrics[
        (run_metrics[f"{gen_col}_acc"] > 0.99) & (run_metrics["input_acc"] > 0.99)
    ]
    if run_metrics.empty:
        return

    # --- 打印性能耗时数据 ---
    print("---执行时间 (Execution time)---")
    # CodeNet 数据集自身报告的时间（如果有这些列）
    print(
        f"[CodeNet 报告] 原程序时间 (ms): {mean_std(run_metrics, 'cpu_time_v0')}"
    )
    print(
        f"[CodeNet 报告] 参考(输出)程序时间 (ms): {mean_std(run_metrics, 'cpu_time_v1')}"
    )

    print("-" * 80)
    # 本次脚本实测获得的时间
    print(f"[本次实测] 原程序时间 (ms): {mean_std(run_metrics, 'input_time')}")
    print(
        f"[本次实测] 参考(输出)程序时间 (ms): {mean_std(run_metrics, 'reference_time')}"
    )
    print(
        f"[本次实测] 模型生成程序时间 (ms): {mean_std(run_metrics, f'{gen_col}_time')}"
    )

    # 找到模型生成的代码确实比参考代码还快的部分
    run_metrics_improved = run_metrics[
        run_metrics[f"{gen_col}_time_mean"] < run_metrics["reference_time_mean"]
    ]
    if len(run_metrics_improved) > 0:
        print("----发生性能改善时的指标 (Metrics when improved)--")
        print(
            f"找到了 {len(run_metrics_improved)} 个模型生成代码比参考代码更快的例子"
        )
        print(
            f"[本次实测] 原程序时间 (ms): {mean_std(run_metrics_improved, 'input_time')}"
        )
        print(
            f"[本次实测] 参考(输出)程序时间 (ms): {mean_std(run_metrics_improved, 'reference_time')}"
        )
        print(
            f"[本次实测] 模型生成程序时间 (ms): {mean_std(run_metrics_improved, f'{gen_col}_time')}"
        )
        
    # 分析异常数据（参考代码测出来比慢代码还慢的情况）
    print(
        f"参考代码实测运行更慢的异常用例数: {len(get_anomalies(run_metrics))}"
    )
    print("----- 其他附加指标 (Additional Metrics) -----")

    # 有效样本：生成代码和输入代码都完全通过全部测试用例
    valid_samples = run_metrics[
        (run_metrics[f"{gen_col}_acc"] == 1.0) & (run_metrics["input_acc"] == 1.0)
        ]
    print(f"有效测试样本数量 (准确率=100%): {len(valid_samples)}")

    # 优化成功：模型生成代码时间比起原较慢的代码降低了至少10% （*1.1 防止计时误差）
    optimized_samples = valid_samples[
        valid_samples[f"{gen_col}_time_mean"] *1.1  < valid_samples["input_time_mean"]
        ]
    print(f"成功被优化的样本数 (比原慢代码更快): {len(optimized_samples)}")


def mean_std(df, col) -> str:
    """辅助函数：计算指定列的平均值和标准差并格式化为字符串。"""
    mean_col = f"{col}_mean"
    std_col = f"{col}_std"
    # 如果找不到以 _mean 结尾的列，说明传入的是普通的单列数据，直接对列算 mean 和 std
    if mean_col not in df.columns or std_col not in df.columns:
        return f"{df[col].mean():.4f} ± {df[col].std():.4f}"

    # 如果有预先计算的均值列和标准差列（在 run_programs 里计算好的），则直接求它们的均值
    return f"{df[mean_col].mean():.4f} ± {df[std_col].mean():.4f}"


def get_anomalies(run_metrics):
    """获取数据异常的案例。
    异常定义：在原始的数据集记录(CodeNet)中被标记为优化并变快了，但在本次沙盒环境实际测量中，参考代码运行时间却比慢代码更长的现象。
    """
    # 计算 CodeNet 数据集里宣称的相对性能提升百分比
    run_metrics["codenet_reported_rel_improvement"] = (
        run_metrics["cpu_time_v0"] - run_metrics["cpu_time_v1"]
    ) / run_metrics["cpu_time_v0"]
    run_metrics["codenet_reported_rel_improvement"] = run_metrics[
        "codenet_reported_rel_improvement"
    ].apply(lambda x: round(x * 100, 2))
    
    # 计算本次沙盒实测获得的相对性能提升百分比
    run_metrics["measured_rel_improvement"] = (
        run_metrics["input_time_mean"] - run_metrics["reference_time_mean"]
    ) / run_metrics["input_time_mean"]
    run_metrics["measured_rel_improvement"] = run_metrics["measured_rel_improvement"].apply(
        lambda x: round(x * 100, 2)
    )
    
    # 标注异常状态
    run_metrics["is_anomaly"] = run_metrics.apply(
        lambda x: x["codenet_reported_rel_improvement"] > 10 and x["measured_rel_improvement"] < 0,
        axis=1,
    )
    # 提取并返回异常的 DataFrame
    run_metrics_anomalies = run_metrics[run_metrics["is_anomaly"]]
    return run_metrics_anomalies


def get_input_from_prompt(
    row: pd.Series,
    question_sep: str = "# slower version:",
    answer_sep: str = "# optimized version of the same code:",
) -> str:
    """基于 Prompt 的生成模式，从字符串 prompt 中提取出输入的代码。"""
    
    # 兼容获取 prompt 字符串的逻辑
    if "entire_prompt" in row:
        prompt_str = row["entire_prompt"]
    else:
        prompt_str = row["prompt"] + row["question"]
    prompt_str = prompt_str.replace("\n\n\n\n\n", "")
    
    # 通过指定的分割符（# slower version:, # optimized version...）切割字符串获取真正的代码部分
    return prompt_str.split(question_sep)[-1].split(answer_sep)[0].strip()


if __name__ == "__main__":
    # 解析命令行参数
    args = EvaluationConfig.get_args()
    args.add_argument("--eval_config", type=str, required=False)
    args = args.parse_args()
    
    # 加载配置
    if args.eval_config is not None:
        # 支持从 YAML 文件读取
        evaluation_config = EvaluationConfig.from_yaml(args.eval_config)
    else:
        # 或者从命令行读取
        evaluation_config = EvaluationConfig.from_args(args)

    # 启动评估主流程
    evaluate_generated_outputs(evaluation_config)