# -*- coding: utf-8 -*-
# print(f"### :\n{}")
# #####################################################################################################################🔖💡✅🟨❌

import os
from tqdm import tqdm
import json
import pandas as pd

import statistics
import pprint
import argparse
from code_evaluation_pipeline import evaluate_code_performance_after_replacement


print('\033[0:33m======================= Come on, You can do it! ==============================\033[m')


# #####################################################################################################################🔖💡✅🟨❌
DEBUG = False


TEST_COLUMN_NAMES = [
'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code__Ori_1', 'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code_1',
'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code__Ori_2', 'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code_2', 
'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code__Ori_3', 'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code_3',
'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code__Ori_4', 'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code_4', 
'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code__Ori_5', 'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code_5', 

]






# #####################################################################################################################🔖💡✅🟨❌





# #####################################################################################################################🔖💡✅🟨❌
def get_hyperparameters():

    parser = argparse.ArgumentParser()

    parser.add_argument("--iteration_rounds",       default=-404, type=int)
    parser.add_argument("--ignore_count",       default=1, type=int)
    parser.add_argument("--run_count_excluding_ignore", default=26, type=int)
    parser.add_argument("--dataset_path",     default=r"Code_Data_Table", type=str)
    parser.add_argument("--save_path",     default=r"IO_Completed_Code_Data_Table", type=str)

    args = parser.parse_args()

    return args



# #####################################################################################################################🔖💡✅🟨❌
def set_extra_parameters(args):
    # -----------------------------------------------------------------------------------------------
    global ITERATION_ROUNDS, IGNORE_COUNT, RUN_COUNT_EXCLUDING_IGNORE, DATASET_PATH, SAVE_PATH
    ITERATION_ROUNDS = args.iteration_rounds
    IGNORE_COUNT = args.ignore_count
    RUN_COUNT_EXCLUDING_IGNORE = args.run_count_excluding_ignore
    DATASET_PATH = args.dataset_path
    SAVE_PATH = args.save_path






# #####################################################################################################################🔖💡✅🟨❌
def main():
    args = get_hyperparameters()
    set_extra_parameters(args)
    print(f"✅✅✅======================= Execution Start, Iteration Rounds: {args.iteration_rounds}")
    print(pprint.pformat(vars(args)))
    if os.path.exists(SAVE_PATH) == False:
        multi_dataset_traversal(args)



# #####################################################################################################################🔖💡✅🟨❌
def multi_dataset_traversal(args):

    # -----------------------------
    if SAVE_PATH.endswith('.csv') == False:   # If save path is not a CSV file
        find_current_save_id(SAVE_PATH)
        os.makedirs(f'{SAVE_PATH}', exist_ok=True)

    if DATASET_PATH.endswith('.csv') == False:   # If dataset path is not a CSV file
        df_name_list = os.listdir(DATASET_PATH)
        assert len(df_name_list) == 1, f"There must be only one file in dataset path {DATASET_PATH}"

        df_name = df_name_list[0]  # Get the first filename
        df = pd.read_csv(f'{DATASET_PATH}/{df_name}')        # Read CSV file  
    else:
        df_name = DATASET_PATH.split('/')[-1].split('\\')[-1]
        df = pd.read_csv(f'{DATASET_PATH}')        # Read CSV file

    # -------------------------------------------------
    """ Set Hyperparameters """
    global DATASET, PROGRAMMING_LANGUAGE
    DATASET = 'Peace'
    if ('_Py' in df_name) and ('_Cpp' not in df_name):
        PROGRAMMING_LANGUAGE = 'python'
    elif ('_Cpp' in df_name) and ('_Py' not in df_name):
        PROGRAMMING_LANGUAGE = 'cpp'



    # ==========================================================================================================================================
    print(f"### Using parameters, Dataset:{DATASET}, Language:({PROGRAMMING_LANGUAGE}), ### df_name:{df_name}")
    df = single_df_traversal(df_name, df)

    # -------------------------------------------------------------------
    if ITERATION_ROUNDS == -404:
        os.remove(f'{DATASET_PATH}/{df_name}')  # Delete original dataset file



# #####################################################################################################################❌
def single_df_traversal(df_name, df):

    original_working_dir = os.getcwd()
 
    for column_name in TEST_COLUMN_NAMES:
        if ('_Ori' not in column_name) and ('_log_probs' not in column_name):
            df = traverse_single_column(df, prediction_column_name=column_name, save_name = column_name)

    # Save modified data to a new CSV file
    if 'PIE_Py_' in df_name:
        pie_id = df_name.split('PIE_Py_')[-1].split('_')[0]
    elif 'PIE_Cpp_' in df_name:
        pie_id = df_name.split('PIE_Cpp_')[-1].split('_')[0]
    elif 'DB_Py_' in df_name:
        pie_id = df_name.split('DB_Py_')[-1].split('_')[0]
    elif 'ECCO_Py_' in df_name:
        pie_id = df_name.split('ECCO_Py_')[-1].split('_')[0]
    elif 'Peace_Py_' in df_name:
        pie_id = df_name.split('Peace_Py_')[-1].split('_')[0]
    

    # Create new df file name
    new_pie_id = str(int(pie_id) + 1).zfill(3)
    original_name_prefix = df_name.split('__')[0].replace(pie_id, new_pie_id)

    new_name = f"{original_name_prefix}__IO_Completed.csv"

    # Save modified data to new CSV file
    os.chdir(original_working_dir)
    if SAVE_PATH.endswith('.csv'):
        df.to_csv(f'{SAVE_PATH}', index=False)
    else:
        df.to_csv(f'{SAVE_PATH}/{new_name}', index=False)

    return df



# #################################################################################################################################################❌
def traverse_single_column(df, prediction_column_name, save_name):
    overall_io_pass_rate, io_pass_rate_list, runtime_list, cpu_instructions_list, memory_usage_list, eval_result_list = traverse_function(df, prediction_column_name=prediction_column_name)
    print(f"✅✅✅ Column Name:{prediction_column_name}  ## Overall IO Pass Rate:{overall_io_pass_rate}%")

    result_dict = {}

    # ---------------------------------------------------------------------
    result_dict[f'{save_name}__IO_pass_rate_(%)'] = io_pass_rate_list
    result_dict[f'{save_name}__time(us)'] = runtime_list
    result_dict[f'{save_name}__CPU_instructions'] = cpu_instructions_list
    result_dict[f'{save_name}__Memory_Usage_(MB)'] = memory_usage_list
    result_dict[f'{save_name}__Evaluation_Result'] = eval_result_list

    # Concatenate all at once after loop finishes
    df = pd.concat([df, pd.DataFrame(result_dict, index=df.index)], axis=1)

    return df


# #################################################################################################################################################
def traverse_function(df, prediction_column_name):

    repo_root_list = df['repo_path'].tolist()
    version_list = df['sha'].tolist()
    target_file_list = df['target_file'].tolist()
    target_class_list = df['target_class'].tolist()
    target_func_list = df['target_func'].tolist()
    venv_path_list = df['venv_path'].tolist()
    test_cmd_list = df['test_cmd'].tolist()
    new_func_body_list = df[prediction_column_name].tolist()

    io_pass_rate_list = []
    runtime_list = []
    cpu_instructions_list = []
    memory_usage_list = []
    eval_result_list = []

    
    for i in tqdm(range(len(new_func_body_list))):  # Iterate through each row
        total_eval_data_dict = {}
        total_eval_data_dict["repo_path"] = repo_root_list[i]
        total_eval_data_dict["sha"] = version_list[i]
        total_eval_data_dict["target_file"] = target_file_list[i] 
        total_eval_data_dict["target_class"] = target_class_list[i]
        total_eval_data_dict["target_func"] = target_func_list[i]
        total_eval_data_dict["venv_path"] = venv_path_list[i]
        total_eval_data_dict["test_cmd"] = test_cmd_list[i]
        total_eval_data_dict["after_code"] = new_func_body_list[i].strip()


        
        # ----------------------------------------------------------------------------------------------------------------
        if prediction_column_name == "Slow_Code" or prediction_column_name == "input":
            io_pass_rate, runtime, cpu_instructions, memory_usage, eval_result_dict = io_evaluation_function(total_eval_data_dict, is_slow_code=True, ignore_count=IGNORE_COUNT, run_count_excluding_ignore=RUN_COUNT_EXCLUDING_IGNORE, statistic_metric='mean')
        else:
            io_pass_rate, runtime, cpu_instructions, memory_usage, eval_result_dict = io_evaluation_function(total_eval_data_dict, is_slow_code=False, ignore_count=IGNORE_COUNT, run_count_excluding_ignore=RUN_COUNT_EXCLUDING_IGNORE, statistic_metric='mean')
        

        io_pass_rate_list.append(io_pass_rate)
        runtime_list.append(runtime)
        cpu_instructions_list.append(cpu_instructions)
        memory_usage_list.append(memory_usage)
        eval_result_list.append(eval_result_dict)
        

    # ----------------------------------------------------------------------------------------------------------------
    # if 'Public_IO_unit_tests' in IO_Type or 'Hide_IO_unit_tests' in IO_Type:
    overall_io_pass_rate = round(len([v for v in io_pass_rate_list if v == 1]) / len(io_pass_rate_list) * 100, 2)
    # elif 'Generate_IO' in IO_Type or 'Gen_IO' in IO_Type:
    #     print(f"### io_pass_rate_list：{set(io_pass_rate_list)}")
    #     overall_io_pass_rate = round(statistics.mean(io_pass_rate_list) * 100, 2)

    return overall_io_pass_rate, io_pass_rate_list, runtime_list, cpu_instructions_list, memory_usage_list, eval_result_list


# #####################################################################################################################🔖💡✅🟨"stdout" in code_line or "stderr"
def check_code_function(tested_code_str):
    risk_keywords = ['setrecursionlimit(', 'stack_size(', "if __name__ == '__main__':", ' is not ', ' is ', 'stdout', 'stderr',
             ", file=sys.stdout", ", file=stdout", ", output=sys.stdout", ", output=stdout", ", file=sys.stderr", ", file=stderr", ", output=sys.stderr", ", output=stderr",
             "sys.stdin.readline", "stdin.readline", "sys.stdin.buffer.readline", "stdin.buffer.readline",
             "sys.stdout.write", "stdout.write", "sys.__stdout__.write", "__stdout__.write", "sys.stderr.write", "stderr.write", "sys.__stderr__.write", "__stderr__.write",
             'return sys.stdout.flush()', 
             "IOWrapper(", "FastIO(", "StringIO(", "BytesIO(", "FastStdout(", ".close(", "stdout", "stderr", 
             'open(',
             "threading", "thread", "multiprocessing", "asyncio", "queue.Queue(", "ProcessPoolExecutor", "concurrent", "fork(", "subprocess.run("]
    
    for keyword in risk_keywords:
        if keyword in tested_code_str and 'open(0' not in tested_code_str:
            return keyword
    return True


# #####################################################################################################################🔖💡✅🟨
def io_evaluation_function(total_eval_data_dict, is_slow_code, ignore_count, run_count_excluding_ignore, statistic_metric='mean'):
    """
    eval_result_dict = {"IO_pass_result": io_pass_result_list,
                        "used_time_list": used_time_list,
                        "error_label": 1,
                        "error_info_dict": {},
                        "tested_code": original_tested_code_str,
                        "time_list_unit": 'microseconds_μs = e-6s',
                        "code_output_list": code_output_list}
    """
    
    # 1. Save current path
    current_path = os.getcwd()  # Get current working directory

    # try:
    io_pass_rate_list, cpu_instructions_list, runtime_list, memory_usage_list = evaluate_code_performance_after_replacement( total_eval_data_dict, 
                                                                                                    is_slow_code=is_slow_code, 
                                                                                                    ignore_count=ignore_count, 
                                                                                                    run_count_excluding_ignore=run_count_excluding_ignore,
                                                                                                    )
    # 3. Switch back to original path
    os.chdir(current_path)
    
    eval_result_dict = {
        "IO_Pass_Result": io_pass_rate_list,
        "CPU_Instructions_List": cpu_instructions_list,
        "Execution_Time_List": runtime_list,
        "Memory_Usage_List": memory_usage_list,
        "Time_List_Unit": 'microseconds_μs = e-6s',
    }

    assert len(io_pass_rate_list) == len(runtime_list), f"IO Pass Single List and IO Time Double List length mismatch {len(io_pass_rate_list)} != {len(runtime_list)}"

    io_pass_rate = statistics.mean(io_pass_rate_list)
    assert io_pass_rate == 1 or io_pass_rate == 0, f"IO Pass Rate range error {io_pass_rate}"

    if io_pass_rate == 0:
        return io_pass_rate, 1234567890, 1234567890, 1234567890, eval_result_dict

    # Process IO Pass Rate == 1
    elif io_pass_rate == 1:
        if statistic_metric == 'median':
            current_runtime = statistics.median(runtime_list)
            current_cpu_instructions = statistics.median(cpu_instructions_list)
            current_memory_usage = statistics.median(memory_usage_list)
        elif statistic_metric == 'mean':
            current_runtime = statistics.mean(runtime_list)
            current_cpu_instructions = statistics.mean(cpu_instructions_list)
            current_memory_usage = statistics.mean(memory_usage_list)

        runtime = round(current_runtime, 2)
        cpu_instructions = round(current_cpu_instructions)
        memory_usage = round(current_memory_usage, 6)

        return io_pass_rate, runtime, cpu_instructions, memory_usage, eval_result_dict

    # --------------------------------------------------------------------------------------------
    # """ Mainly execution time is only calculated when IO is 1 """
    else:
        raise NotImplementedError("Currently only handles cases where IO Pass Rate is 0 or 1")

        
# #####################################################################################################################🔖💡✅🟨
def find_current_save_id(save_path_prefix):
    global SAVE_PATH
    history_save_names = os.listdir('./')
    history_ids = []
    for name in history_save_names:
        if name.startswith(SAVE_PATH):
            history_ids.append(int(name.split(SAVE_PATH)[-1]))
    current_id = str(max(history_ids, default=0) + 1).zfill(2)
    SAVE_PATH = f"{save_path_prefix}{current_id}"




# #################################################################################################################################################
if __name__ == '__main__':
    print("✅✅✅############################################################################################################ Start File T8__PIE__Evaluate_Repo_Execution_Time__Latest6.py")
    main()