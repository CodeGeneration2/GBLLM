import os
import subprocess
import tempfile
import json
import re

import pandas as pd  # 新增：导入 pandas 库






# #####################################################################################################################🔖💡✅🟨❌
数据集路径 = r'./PIE_Cpp_008_基础表__更改顺序.csv'  # 修改为你的数据集路径




# #####################################################################################################################🔖💡✅🟨❌
# 主程序入口
def main():  # 修改：将 df 作为参数传入
    
    # 1. 从 JSON 文件加载代码数据集
    # 数据集应该包含未优化代码(source_code)和优化后代码(optimized_code)配对
    # with open('../path/to/dataset', 'r') as f:
    #     df = json.load(f)
    # 假设你的 df 已经在这里准备好了，例如从 csv 或其他地方读取
    df = pd.read_csv(数据集路径)

    处理后数据_列表 = []

    # 2. 遍历数据集中的每一条记录
    # 修改：直接遍历 df 表中的每一行
    for idx, row in df.iterrows():
        id = row['712_idx']
        # 预处理代码（清理头文件和特殊语法）
        source_code = preprocess_code(row['input'])
        optimized_code = preprocess_code(row['target'])

        print(f"Processing entry {idx+1}/{len(df)}: ID {id}")

        # 3. 调用 Clang 提取这两种代码的 CFG 文本输出
        source_cfg_output = get_cfg_output(source_code)
        optimized_cfg_output = get_cfg_output(optimized_code)

        # 4. 容错处理：如果未能成功提取到 CFG，跳过该条目
        if not source_cfg_output.strip():
            print(f"Error: No CFG output for source code of ID {id}")
            continue  
            
        if not optimized_cfg_output.strip():
            print(f"Error: No CFG output for optimized code of ID {id}")
            continue  

        # 5. 将 Clang 的纯文本输出解析成 Python 可操作的数据结构（BasicBlock 字典）
        source_blocks = parse_cfg_output(source_cfg_output)
        optimized_blocks = parse_cfg_output(optimized_cfg_output)

        # 6. 对比这两个控制流图，生成差异标签列表
        labels = compare_cfgs(source_blocks, optimized_blocks)

        # 7. 构建新的数据条目，保留了原始ID、差异标签以及生成的CFG原始文本
        new_entry = {
            'id': id,
            'labels': labels,
            'source_cfg': source_cfg_output,
            'optimized_cfg': optimized_cfg_output
        }

        处理后数据_列表.append(new_entry)

    # 8. 将提取出的差异特征(labels)和CFG数据保存回新的 JSON 文件中
    with open('PIE数据.json', 'w') as f:
        json.dump(处理后数据_列表, f, indent=4)









# #####################################################################################################################🔖💡✅🟨❌
# 定义“基本块”(Basic Block)类，用于表示控制流图(CFG)中的节点
# 在编译器理论中，基本块是只包含一条入口和一条出口的直线代码序列
class BasicBlock:
    def __init__(self, name):
        self.name = name                 # 基本块的名称，例如 'B1', 'B2'
        self.statements = []             # 该基本块中包含的具体代码语句列表
        self.successors = []             # 后继节点列表（执行完该块后可能跳转到的块）
        self.predecessors = []           # 前驱节点列表（哪些块执行完后会跳转到该块）

    def __repr__(self):
        # 定义对象的字符串表示形式，方便调试和打印查看
        return (f"BasicBlock({self.name}, "
                f"statements={self.statements}, "
                f"successors={self.successors}, "
                f"predecessors={self.predecessors})")

# 预处理C++代码字符串，使其更容易被 Clang 静态分析器正确解析
def preprocess_code(code_str):
    # 替换 C++ 中的万能头文件 #include <bits/stdc++.h> 为具体的标准库头文件。
    # 因为 Clang 的静态分析器有时无法正确解析或找不到系统底层的 bits/stdc++.h。
    code_str = re.sub(
        r'#include\s*<bits/stdc\+\+\.h>',
        '''
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <map>
#include <set>
#include <queue>
#include <stack>
#include <deque>
#include <list>
#include <functional>
#include <numeric>
#include <utility>
#include <limits>
#include <iomanip>  // 用于 setprecision
using namespace std;
''',
        code_str
    )

    # 移除 C++ 中的 'register' 关键字（C++17 已经废弃此关键字，Clang 解析时可能会报错）
    code_str = re.sub(r'\bregister\b', '', code_str)

    # 确保如果代码中使用了 'setprecision'（格式化输出），则引入 '<iomanip>' 头文件
    if 'setprecision' in code_str and '<iomanip>' not in code_str:
        code_str = code_str.replace('#include <iostream>', '#include <iostream>\n#include <iomanip>')

    # 移除 GCC 编译器特有的扩展属性 __attribute__((...))，因为 Clang 可能无法识别这些特定的 GCC 扩展
    code_str = re.sub(r'__attribute__\s*\(\(.*?\)\)', '', code_str)

    return code_str



# #####################################################################################################################🔖💡✅🟨❌
# 使用 Clang 的静态分析器提取 C++ 代码的控制流图 (CFG) 输出
def get_cfg_output(code_str):
    """
    Use Clang's static analyzer to dump CFGs.
    """
    # 将代码字符串保存到一个临时的 .cpp 文件中，以便 Clang 读取
    # delete=False 表示关闭文件后不立即删除，我们需要稍后手动清理
    with tempfile.NamedTemporaryFile(delete=False, suffix='.cpp', mode='w', encoding='utf-8') as temp_file:
        temp_file.write(code_str)
        temp_filename = temp_file.name

    # 构造 Clang 命令行指令
    clang_cmd = [
        # r'D:\Clang__LLVM\bin\clang++.exe',
        'clang++',
        '-std=c++14',                     # 使用 C++14 标准
        '-w',                             # 屏蔽所有警告信息，保持输出干净
        '-Xclang', '-analyze',            # 调用 Clang 的内部静态分析器
        '-Xclang', '-analyzer-checker=debug.DumpCFG', # 开启 DumpCFG 检查器，用于打印 CFG
        '-fsyntax-only',                  # 只进行语法检查和分析，不进行实际的编译和链接
        temp_filename                     # 目标临时文件
    ]
    try:
        # 执行 Clang 命令并捕获控制台输出
        result = subprocess.run(clang_cmd, capture_output=True, text=True)
        cfg_output = result.stdout + result.stderr  # 合并标准输出和错误输出（CFG 通常打印在 stderr 或 stdout 中）
        
        # 如果命令执行失败（返回码不为0），打印错误信息并将输出置空
        if result.returncode != 0:
            print(f"Clang returned non-zero exit code {result.returncode}")
            print(f"Clang error output:\n{cfg_output}")
            cfg_output = ''  # 如果 Clang 失败，丢弃输出
    except Exception as e:
        # 捕获类似找不到 clang++ 命令等系统级异常
        print(f"Error running Clang: {e}")
        cfg_output = ''
    finally:
        # 无论成功或失败，都清理掉之前创建的临时 .cpp 文件
        os.unlink(temp_filename)

    return cfg_output



# #####################################################################################################################🔖💡✅🟨❌
# 解析 Clang 输出的 CFG 文本，将其转化为 BasicBlock 对象的字典
def parse_cfg_output(cfg_output):
    blocks = {}
    current_block = None

    # 按行读取 Clang 的输出
    lines = cfg_output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('***'):
            continue  # 跳过分隔符行（例如 **********）
        elif line.startswith('CFG for'):
            continue  # 跳过表头信息（例如 CFG for function 'main'）
        elif re.match(r'^\[B\d+( \(.*\))?\]', line):
            # 匹配基本块的标题行，例如 [B1] 或 [B2 (NORETURN)]
            block_name = line.split()[0].strip('[]') # 提取 'B1' 这样的名字
            current_block = BasicBlock(block_name)   # 创建新的 BasicBlock 对象
            blocks[block_name] = current_block       # 存入字典中
        elif current_block is not None:
            # 如果当前正在解析某个基本块内部
            if line.startswith('Preds') or line.startswith('Succs'):
                # 匹配前驱 (Preds) 或后继 (Succs) 节点行
                key, rest = line.split(':', 1)
                # 使用正则找出所有形如 [BX] 的引用
                block_refs = re.findall(r'\[B\d+( \(.*\))?\]', rest)
                block_names = [ref.strip('[]').split()[0] for ref in block_refs]
                
                if key.startswith('Preds'):
                    current_block.predecessors = block_names
                elif key.startswith('Succs'):
                    current_block.successors = block_names
            else:
                # 如果不是前驱/后继，则说明是具体的代码语句，将其追加到语句列表中
                current_block.statements.append(line)
        else:
            continue  # 忽略在块外部的其他无用行

    return blocks



# #####################################################################################################################🔖💡✅🟨❌
# 比较两个控制流图（未优化的 vs 优化后的），提取出差异标签(labels)
def compare_cfgs(blocks1, blocks2):
    labels = []
    differences_found = False

    # 获取两个 CFG 中所有基本块名称的并集，并排序以保证比较顺序的一致性
    all_block_names = set(blocks1.keys()).union(set(blocks2.keys()))

    for block_name in sorted(all_block_names):
        block1 = blocks1.get(block_name)
        block2 = blocks2.get(block_name)

        # 情况1：块在优化前存在，但优化后被删除了
        if block1 and not block2:
            labels.append(f"Block {block_name} removed in optimized code")
            differences_found = True
            continue
            
        # 情况2：块在优化前不存在，但优化后新增了
        if block2 and not block1:
            labels.append(f"Block {block_name} added in optimized code")
            differences_found = True
            continue

        # 此时说明该块在两边都存在，比较块内的语句
        statements1 = block1.statements
        statements2 = block2.statements

        if statements1 != statements2:
            differences_found = True
            labels.append(f"Block {block_name} statements changed")
            # 逐行对比具体的语句差异
            for idx in range(max(len(statements1), len(statements2))):
                stmt1 = statements1[idx] if idx < len(statements1) else "<no statement>"
                stmt2 = statements2[idx] if idx < len(statements2) else "<no statement>"
                if stmt1 != stmt2:
                    labels.append(f"Block {block_name}, statement {idx+1} changed from '{stmt1}' to '{stmt2}'")
            # 记录语句数量的变化
            if len(statements1) != len(statements2):
                labels.append(f"Block {block_name} statement count changed from {len(statements1)} to {len(statements2)}")

        # 比较后继节点的差异（控制流跳转发生了变化）
        if block1.successors != block2.successors:
            differences_found = True
            labels.append(f"Block {block_name} successors changed from {block1.successors} to {block2.successors}")

        # 比较前驱节点的差异
        if block1.predecessors != block2.predecessors:
            differences_found = True
            labels.append(f"Block {block_name} predecessors changed from {block1.predecessors} to {block2.predecessors}")

    # 如果遍历完毕没有任何不同，打上无差异标签
    if not differences_found:
        labels.append("No differences detected between source and optimized CFGs")

    return labels





# #####################################################################################################################🔖💡✅🟨❌
if __name__ == "__main__":
    main()