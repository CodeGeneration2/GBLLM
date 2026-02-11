# -*- coding: utf-8 -*-
# print(f"### :\n{}")
# #####################################################################################################################🔖💡✅🟨

import sys
import re
import os
from tqdm import tqdm
import io
import tokenize
import pandas as pd
import pprint
import statistics
from collections import Counter

print('\033[0:33m=======================Go, Go, Go! You can do it! ==============================\033[m')

# #####################################################################################################################🔖💡✅🟨
DEBUG = False
PRINT_ERROR_INFO = False
SAVE_DICT = False
DATASET_PATH  = r""


# --------------------------------------
HISTORY_EDIT_COLUMN_PREFIXES = [
    'SBLLM_cot_G5',
    ]
COLUMN_PREFIX = "Ablation_Remove_NL_Cot_CFG_SlowMidFastTime_Round1_G5"
COLUMN_PREFIX = "Ablation_Remove_IO_Cot_NL_CFG_SlowMidFastTime_Round1_G5"
COLUMN_PREFIX = "Ablation_Remove_CFG_Cot_NL_SlowMidFastTime_Round1_G5"
COLUMN_PREFIX = "Ablation_All_CFG_Cot_NL_SlowMidFastTime_Round1_G5"
COLUMN_PREFIX = "Ablation_Remove_Time_Cot_NL_SlowMidFast_Round1_G5"
COLUMN_PREFIX = "Ablation_Remove_Trajectory_Cot_NL_Round1_G5"
COLUMN_PREFIX = "Ablation_Remove_All_Cot_Round1_G5"






# --------------------------------------
STANDARD_CORRECT_IO_THRESHOLD = 0.999
PREDICTED_CORRECT_IO_THRESHOLD = 0.999

SLOW_CODE_NAME = "input"



# #####################################################################################################################🔖💡✅🟨
def main(mode = 0):
    correct_ratio_list = []
    optimization_ratio_list = []
    speedup_ratio_list = []

    # Comprehensive evaluation, 9 types of results   
    if mode == 0:
        for i in range(1, 6):
            correct_ratio, optimization_ratio, speedup_ratio = evaluate_generate1_top_1_func(i)
            correct_ratio_list.append(str(correct_ratio))
            optimization_ratio_list.append(str(optimization_ratio))
            speedup_ratio_list.append(str(speedup_ratio))
        for (index, (generation_count, submission_count)) in enumerate([(3, 3), (5, 5), (3, 1), (5, 1), (5, 3)]):
            correct_ratio, optimization_ratio, speedup_ratio = statistics_generate_n_top_k_func(generation_count, submission_count)
            correct_ratio_list.append(str(correct_ratio))
            optimization_ratio_list.append(str(optimization_ratio))
            speedup_ratio_list.append(str(speedup_ratio))
            if index == 1:
                correct_ratio_list.append("")
                optimization_ratio_list.append("")
                speedup_ratio_list.append("")
        
        # ------------------------------------------------------------------------------
        if ("Cot_NL_CFG_SlowMidFast" in COLUMN_PREFIX) or ("Ablation" in COLUMN_PREFIX):
            round_num = int(COLUMN_PREFIX.split("_Round")[1].split("_G5")[0])
            correct_ratio_list.append("")
            optimization_ratio_list.append("")
            speedup_ratio_list.append("")
            if ("Ablation" in COLUMN_PREFIX):
                HISTORY_EDIT_COLUMN_PREFIXES.append(COLUMN_PREFIX)
            for submission_count in [1, 3, 5]:
                correct_ratio, optimization_ratio, speedup_ratio = statistics_history_edit_library_generate_n_top_k_func(submission_count, round_num)
                correct_ratio_list.append(str(correct_ratio))
                optimization_ratio_list.append(str(optimization_ratio))
                speedup_ratio_list.append(str(speedup_ratio))

    # Evaluate Repository Level
    elif mode == 3:
        round_num = int(COLUMN_PREFIX.split("_Round")[1].split("_G5")[0])
        correct_ratio, optimization_ratio, speedup_ratio = statistics_repo_level_history_edit_library_generate_n_top_k_func(submission_count=1, round_num=round_num)
        correct_ratio_list.append(str(correct_ratio))
        optimization_ratio_list.append(str(optimization_ratio))
        speedup_ratio_list.append(str(speedup_ratio))

    # Evaluate original efficient code from dataset, i.e., "target" "Fast_code"
    elif mode == 4:
        correct_ratio, optimization_ratio, speedup_ratio = evaluate_generate1_top_1_func(-2)
        correct_ratio_list.append(str(correct_ratio))
        optimization_ratio_list.append(str(optimization_ratio))
        speedup_ratio_list.append(str(speedup_ratio))

    # Single result
    elif mode == 5:
        correct_ratio, optimization_ratio, speedup_ratio = evaluate_generate1_top_1_func()
        correct_ratio_list.append(str(correct_ratio))
        optimization_ratio_list.append(str(optimization_ratio))
        speedup_ratio_list.append(str(speedup_ratio))

    # Evaluate results for each n
    elif mode == 6:
        for i in range(1, 6):
            correct_ratio, optimization_ratio, speedup_ratio = evaluate_generate1_top_1_func(i)
            correct_ratio_list.append(str(correct_ratio))
            optimization_ratio_list.append(str(optimization_ratio))
            speedup_ratio_list.append(str(speedup_ratio))

    # Use Public I/O method, Evaluate Top_k results    
    elif mode == 7:
        for i in [1, 3, 5]:
            correct_ratio, optimization_ratio, speedup_ratio = statistics_generate_n_top_k_func(generation_count, submission_count = i)
            correct_ratio_list.append(str(correct_ratio))
            optimization_ratio_list.append(str(optimization_ratio))
            speedup_ratio_list.append(str(speedup_ratio))

    # Use Generated I/O method, Evaluate Top_k results    
    elif mode == 8:
        for i in [1, 3, 5]:
            correct_ratio, optimization_ratio, speedup_ratio = statistics_generate_io_filtering_top_k_func(Top_k = i)
            correct_ratio_list.append(str(correct_ratio))
            optimization_ratio_list.append(str(optimization_ratio))
            speedup_ratio_list.append(str(speedup_ratio))

    
    # Based on max probability. When generating n models, ChatGPT saves sorted by probability.
    elif mode == 10:
        evaluate_generate1_top_1_func(1)
        statistics_generate_n_top_k_probability_filtering_func(Top_k = 3)

    # Filter Top_k probability filtering
    elif mode == 21:
        evaluate_generate1_top_1_func(1)
        statistics_generate_n_top_k_probability_filtering_func(Top_k = 3)

    
    # ======================================================================================
    print(f"\n======================= ### Correct Ratio List, Optimization Ratio List, Speedup Ratio List: ")
    # print('\n'.join(correct_ratio_list))
    # print('\n'.join(optimization_ratio_list))
    # print('\n'.join(speedup_ratio_list))
    with open(r"C:\Users\15121\Desktop\data_results.txt", "w", encoding="utf-8") as f:
        for i in range(len(correct_ratio_list)):
            f.write(f"{correct_ratio_list[i]}\t{optimization_ratio_list[i]}\t{speedup_ratio_list[i]}\n")
            print(f"{correct_ratio_list[i]}\t{optimization_ratio_list[i]}\t{speedup_ratio_list[i]}")





# #####################################################################################################################🔖💡✅🟨
def evaluate_generate1_top_1_func(i = -1):
    # Read CSV file  
    df = pd.read_csv(f'{DATASET_PATH}')

    if '_Py39' in COLUMN_PREFIX:
        original_slow_code_io_list = df['input_Py39__IO_pass_rate_(%)'].tolist()
        original_slow_code_time_list = df['input_Py39__time(ms)'].tolist()
    elif '_Py310' in COLUMN_PREFIX:
        original_slow_code_io_list = df['input_Py310__IO_pass_rate_(%)'].tolist()
        original_slow_code_time_list = df['input_Py310__time(ms)'].tolist()
    else:
        original_slow_code_io_list = df['input__IO_pass_rate_(%)'].tolist()
        original_slow_code_time_list = df['input__time(ms)'].tolist()

    if i > 0:
        generated_code_io_rate_list = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{i}__IO_pass_rate_(%)'].tolist()
        generated_code_time_list = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{i}__time(ms)'].tolist()
    elif i == -1:
        generated_code_io_rate_list = df[f'{COLUMN_PREFIX}__Predict_Fast_code__IO_pass_rate_(%)'].tolist()
        generated_code_time_list = df[f'{COLUMN_PREFIX}__Predict_Fast_code__time(ms)'].tolist()
    elif i == -2 and ('_Py39' in COLUMN_PREFIX):
        generated_code_io_rate_list = df[f'target_Py39__IO_pass_rate_(%)'].tolist()
        generated_code_time_list = df[f'target_Py39__time(ms)'].tolist()
    elif i == -2 and ('_Py310' in COLUMN_PREFIX):
        generated_code_io_rate_list = df[f'target_Py310__IO_pass_rate_(%)'].tolist()
        generated_code_time_list = df[f'target_Py310__time(ms)'].tolist()
    elif i == -2:
        generated_code_io_rate_list = df[f'target__IO_pass_rate_(%)'].tolist()
        generated_code_time_list = df[f'target__time(ms)'].tolist()

    if "DB_Py_" not in DATASET_PATH:
        assert average_io_accuracy_func(original_slow_code_io_list) == 100, "### Original slow code IO pass rate is not 100! "
    average_io_accuracy = average_io_accuracy_func(generated_code_io_rate_list)
    print(f"\n### {COLUMN_PREFIX}_list, Average IO Pass Rate: {average_io_accuracy}")

    print(f"### Original Slow Code Time List, {COLUMN_PREFIX}_list")
    optimization_ratio, speedup_ratio = output_optimization_metrics(original_slow_code_io_list, generated_code_io_rate_list, original_slow_code_time_list, generated_code_time_list)


    if PRINT_ERROR_INFO:
        if i > 0:
            analyze_code_error_types_func(df[f'{COLUMN_PREFIX}__Predict_Fast_code_{i}__Eval_Result'].tolist())
        elif i == -1:
            analyze_code_error_types_func(df[f'{COLUMN_PREFIX}__Predict_Fast_code__Eval_Result'].tolist())
        elif i == -2:
            analyze_code_error_types_func(df[f'input__Eval_Result'].tolist())
            analyze_code_error_types_func(df[f'target__Eval_Result'].tolist())

    # -------------------------------------------------------------------------
    if (SAVE_DICT) and (i == 1) and ('GBLLM' not in DATASET_PATH):
        human_io_pass_rate_list   = df[f'target__IO_pass_rate_(%)'].tolist()
        human_io_execution_time_list = df[f'target__time(ms)'].tolist()
        save_result_dict = { "original_slow_code_pass_io_list": original_slow_code_io_list,
                        "original_slow_code_time_list": original_slow_code_time_list,
                        "human_io_pass_rate_list": human_io_pass_rate_list,
                        "human_execution_time_list": human_io_execution_time_list,
                        "fast_code_io_pass_list": generated_code_io_rate_list,
                        "fast_code_time_list": generated_code_time_list, }
        method_name = COLUMN_PREFIX.split("SBLLM_")[-1].split("_")[0]
        save_path = DATASET_PATH.split("\\")[-1].split("__")[0] + "_" + method_name + f"_submit1.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(str(save_result_dict))
        print(f"\n✅✅✅ Result dictionary saved to: {save_path}")

    return average_io_accuracy, optimization_ratio, speedup_ratio






# #####################################################################################################################🔖💡✅🟨
def statistics_generate_n_top_k_func(generation_count, submission_count, ablation_use_max_probability=False):

    """
    4 double lists. First, I use the public IO pass rate greater than 0.99, then I select the index with the minimum public IO execution time,
    and then collect the private IO pass rate and execution time at the corresponding index:
    """
    # Read CSV file  
    df = pd.read_csv(f'{DATASET_PATH}')
    
    public_io_pass_double_list = [[] for _ in range(generation_count)]
    public_io_time_double_list = [[] for _ in range(generation_count)]
    private_io_pass_double_list = [[] for _ in range(generation_count)]
    private_io_time_double_list = [[] for _ in range(generation_count)]
    for code_idx in range(generation_count):
        public_io_pass_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__Public_IO_pass_rate_(%)'].tolist()
        public_io_time_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__Public_time(ms)'].tolist()
        private_io_pass_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__IO_pass_rate_(%)'].tolist()
        private_io_time_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__time(ms)'].tolist()
        
    assert type(public_io_pass_double_list[0][0]) == type(public_io_time_double_list[0][0]) == type(private_io_pass_double_list[0][0]) == type(private_io_time_double_list[0][0]) == float, "### Not float type! "

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    if generation_count == submission_count:
        selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_private_io_evaluate_top_generate_n_submit_n_func(generation_count, submission_count, private_io_pass_double_list, private_io_time_double_list)
    elif submission_count == 1:
        selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_public_io_evaluate_top_submit_1_func(generation_count, public_io_pass_double_list, public_io_time_double_list, private_io_pass_double_list, private_io_time_double_list)
    elif generation_count == 5 and submission_count == 3:
        selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_public_io_evaluate_top_5_select_3_func(public_io_pass_double_list, public_io_time_double_list, private_io_pass_double_list, private_io_time_double_list)
    else:
        raise ValueError("### Generation count must be greater than or equal to submission count! ")
    


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    original_slow_code_io_list = df[f'{SLOW_CODE_NAME}__IO_pass_rate_(%)'].tolist()
    original_slow_code_time_list = df[f'{SLOW_CODE_NAME}__time(ms)'].tolist()
    assert len(original_slow_code_io_list) == len(selected_extreme_io_pass_rate_list) == len(original_slow_code_time_list) == len(selected_min_execution_time_list), "### Length mismatch! "
    if "DB_Py_" not in DATASET_PATH:
        assert average_io_accuracy_func(original_slow_code_io_list) == 100, "### Original slow code IO pass rate is not 100! "

    # --------------------------------------------------------------------------------------------------------------------------------
    average_io_accuracy = average_io_accuracy_func(selected_extreme_io_pass_rate_list)
    print(f"\n### {COLUMN_PREFIX}, 5 @ {submission_count}, Average IO Pass Rate: {average_io_accuracy}")

    print(f"### Original Slow Code Time List, {COLUMN_PREFIX}_list, 5 @ {submission_count}:")
    optimization_ratio, speedup_ratio = output_optimization_metrics(original_slow_code_io_list, selected_extreme_io_pass_rate_list, original_slow_code_time_list, selected_min_execution_time_list)

    return average_io_accuracy, optimization_ratio, speedup_ratio





# #####################################################################################################################🔖💡✅🟨
def statistics_history_edit_library_generate_n_top_k_func(submission_count, round_num):

    """
    4 double lists. First, I use the public IO pass rate greater than 0.99, then I select the index with the minimum public IO execution time,
    and then collect the private IO pass rate and execution time at the corresponding index:
    """
    # Read CSV file  
    df = pd.read_csv(f'{DATASET_PATH}')
    generation_count = (round_num+1) * 5
    used_code_count_n = 5
    public_io_pass_double_list = []
    public_io_time_double_list = []
    private_io_pass_double_list = []
    private_io_time_double_list = []
    for code_idx in range(5):
        public_io_pass_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[0]}__Predict_Fast_code_{code_idx+1}__Public_IO_pass_rate_(%)'].tolist())
        public_io_time_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[0]}__Predict_Fast_code_{code_idx+1}__Public_time(ms)'].tolist())
        private_io_pass_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[0]}__Predict_Fast_code_{code_idx+1}__IO_pass_rate_(%)'].tolist())
        private_io_time_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[0]}__Predict_Fast_code_{code_idx+1}__time(ms)'].tolist())
    for current_round in range(1, round_num+1):
        if current_round == round_num:
            used_code_count_n = submission_count
            generation_count = generation_count - (5 - submission_count)
        for code_idx in range(used_code_count_n):
            public_io_pass_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[current_round]}__Predict_Fast_code_{code_idx+1}__Public_IO_pass_rate_(%)'].tolist())
            public_io_time_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[current_round]}__Predict_Fast_code_{code_idx+1}__Public_time(ms)'].tolist())
            private_io_pass_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[current_round]}__Predict_Fast_code_{code_idx+1}__IO_pass_rate_(%)'].tolist())
            private_io_time_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[current_round]}__Predict_Fast_code_{code_idx+1}__time(ms)'].tolist())

    assert type(public_io_pass_double_list[0][0]) == type(public_io_time_double_list[0][0]) == type(private_io_pass_double_list[0][0]) == type(private_io_time_double_list[0][0]) == float, "### Not float type! "
    assert len(public_io_pass_double_list) == len(public_io_time_double_list) == len(private_io_pass_double_list) == len(private_io_time_double_list) == generation_count, "### Length mismatch! "

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    if submission_count == 1:
        selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_public_io_evaluate_top_submit_1_func(generation_count, public_io_pass_double_list, public_io_time_double_list, private_io_pass_double_list, private_io_time_double_list)
    elif (submission_count == 3):
        selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_public_io_evaluate_top_5_select_3_func(public_io_pass_double_list, public_io_time_double_list, private_io_pass_double_list, private_io_time_double_list, generation_count=generation_count, submission_count=submission_count)
    elif (submission_count == 5):
        selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_public_io_evaluate_top_5_select_3_func(public_io_pass_double_list, public_io_time_double_list, private_io_pass_double_list, private_io_time_double_list, generation_count=generation_count, submission_count=submission_count)
    else:
        raise ValueError("### Generation count must be greater than or equal to submission count! ")
    


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    original_slow_code_io_list = df[f'{SLOW_CODE_NAME}__IO_pass_rate_(%)'].tolist()
    original_slow_code_time_list = df[f'{SLOW_CODE_NAME}__time(ms)'].tolist()
    assert len(original_slow_code_io_list) == len(selected_extreme_io_pass_rate_list) == len(original_slow_code_time_list) == len(selected_min_execution_time_list), "### Length mismatch! "
    if "DB_Py_" not in DATASET_PATH:
        assert average_io_accuracy_func(original_slow_code_io_list) == 100, "### Original slow code IO pass rate is not 100! "

    # --------------------------------------------------------------------------------------------------------------------------------
    average_io_accuracy = average_io_accuracy_func(selected_extreme_io_pass_rate_list)
    print(f"\n### {COLUMN_PREFIX}, 5 @ {submission_count}, Average IO Pass Rate: {average_io_accuracy}")

    print(f"### Original Slow Code Time List, {COLUMN_PREFIX}_list, 5 @ {submission_count}:")
    optimization_ratio, speedup_ratio = output_optimization_metrics(original_slow_code_io_list, selected_extreme_io_pass_rate_list, original_slow_code_time_list, selected_min_execution_time_list)

    # -------------------------------------------------------------------------
    if SAVE_DICT and submission_count == 1:
        human_io_pass_rate_list   = df[f'target__IO_pass_rate_(%)'].tolist()
        human_io_execution_time_list = df[f'target__time(ms)'].tolist()
        save_result_dict = { "original_slow_code_pass_io_list": original_slow_code_io_list,
                        "original_slow_code_time_list": original_slow_code_time_list,
                        "human_io_pass_rate_list": human_io_pass_rate_list,
                        "human_execution_time_list": human_io_execution_time_list,
                        "fast_code_io_pass_list": selected_extreme_io_pass_rate_list,
                        "fast_code_time_list": selected_min_execution_time_list, }
        save_path = DATASET_PATH.split("\\")[-1].split("__")[0] + f"_GBLLM_submit{submission_count}.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(str(save_result_dict))
        print(f"\n✅✅✅ Result dictionary saved to: {save_path}")


    return average_io_accuracy, optimization_ratio, speedup_ratio




# #####################################################################################################################🔖💡✅🟨
def statistics_repo_level_history_edit_library_generate_n_top_k_func(submission_count, round_num):
    """
    4 double lists. First, I use the public IO pass rate greater than 0.99, then I select the index with the minimum public IO execution time,
    and then collect the private IO pass rate and execution time at the corresponding index:
    """
    # Read CSV file  
    df = pd.read_csv(f'{DATASET_PATH}')
    generation_count = (round_num+1) * 5
    used_code_count_n = 5
    private_io_pass_double_list = []
    private_io_time_double_list = []
    for current_round in range(1, round_num+1):
        if current_round == round_num:
            used_code_count_n = submission_count
            generation_count = generation_count - (5 - submission_count)
        for code_idx in range(used_code_count_n):
            private_io_pass_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[current_round]}__Predict_Fast_code_{code_idx+1}__IO_pass_rate_(%)'].tolist())
            private_io_time_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[current_round]}__Predict_Fast_code_{code_idx+1}__time(us)'].tolist())

    assert type(private_io_pass_double_list[0][0]) == type(private_io_time_double_list[0][0]) == float, "### Not float type! "
    assert len(private_io_pass_double_list) == len(private_io_time_double_list) == generation_count, "### Length mismatch! "

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_private_io_evaluate_top_generate_n_submit_n_func(generation_count, submission_count, private_io_pass_double_list, private_io_time_double_list)
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    original_slow_code_io_list = df[f'{SLOW_CODE_NAME}__IO_pass_rate_(%)'].tolist()
    original_slow_code_time_list = df[f'{SLOW_CODE_NAME}__time(us)'].tolist()
    assert len(original_slow_code_io_list) == len(selected_extreme_io_pass_rate_list) == len(original_slow_code_time_list) == len(selected_min_execution_time_list), "### Length mismatch! "
    if "DB_Py_" not in DATASET_PATH:
        assert average_io_accuracy_func(original_slow_code_io_list) == 100, "### Original slow code IO pass rate is not 100! "

    # --------------------------------------------------------------------------------------------------------------------------------
    average_io_accuracy = average_io_accuracy_func(selected_extreme_io_pass_rate_list)
    print(f"\n### {COLUMN_PREFIX}, 5 @ {submission_count}, Average IO Pass Rate: {average_io_accuracy}")

    print(f"### Original Slow Code Time List, {COLUMN_PREFIX}_list, 5 @ {submission_count}:")
    optimization_ratio, speedup_ratio = output_optimization_metrics(original_slow_code_io_list, selected_extreme_io_pass_rate_list, original_slow_code_time_list, selected_min_execution_time_list)

    return average_io_accuracy, optimization_ratio, speedup_ratio




# #####################################################################################################################🔖💡✅🟨
def use_public_io_evaluate_top_submit_1_func(generation_count, public_io_pass_double_list, public_io_time_double_list, private_io_pass_double_list, private_io_time_double_list):
    """
    4 double lists. First, I use the public IO pass rate greater than 0.99, then I select the index with the minimum public IO execution time,
    and then collect the private IO pass rate and execution time at the corresponding index:
    """

    selected_extreme_io_pass_rate_list = []
    selected_min_execution_time_list = []
    for pie_index in range(len(private_io_pass_double_list[0])):
        # Step 1: Find candidate indices in public IO pass rate greater than 0.99
        candidate_indices_list = [n for n in range(generation_count) if public_io_pass_double_list[n][pie_index] >= PREDICTED_CORRECT_IO_THRESHOLD]
        if candidate_indices_list == []:
            candidate_indices_list = [n for n in range(generation_count) if public_io_pass_double_list[n][pie_index] >= 0.05]
        
        # Step 2: If candidate indices list is not empty, select the index corresponding to the minimum public IO execution time
        if candidate_indices_list:
            extreme_index = min(candidate_indices_list, key=lambda idx: public_io_time_double_list[idx][pie_index])
        else:
            # If there are no candidate values (i.e., no public IO pass rate greater than 0.99), default to 0 or use other strategies as needed
            # print(f"### No candidate values (i.e., no public IO pass rate greater than 0.99), using default index: {chosen_index}")
            extreme_index = 0

        # Extract the corresponding IO pass rate and execution time based on the selected index
        selected_extreme_io_pass_rate_list.append(private_io_pass_double_list[extreme_index][pie_index])
        selected_min_execution_time_list.append(private_io_time_double_list[extreme_index][pie_index])

    return selected_extreme_io_pass_rate_list, selected_min_execution_time_list



# #####################################################################################################################🔖💡✅🟨
def repo_multi_submit_1_func(generation_count, private_io_pass_double_list, private_io_time_double_list):
    """
    4 double lists. First, I use the public IO pass rate greater than 0.99, then I select the index with the minimum public IO execution time,
    and then collect the private IO pass rate and execution time at the corresponding index:
    """

    selected_extreme_io_pass_rate_list = []
    selected_min_execution_time_list = []
    for pie_index in range(len(private_io_pass_double_list[0])):
        # Step 1: Find candidate indices in public IO pass rate greater than 0.99
        candidate_indices_list = [n for n in range(generation_count) if private_io_pass_double_list[n][pie_index] >= PREDICTED_CORRECT_IO_THRESHOLD]
        if candidate_indices_list == []:
            candidate_indices_list = [n for n in range(generation_count) if public_io_pass_double_list[n][pie_index] >= 0.05]
        
        # Step 2: If candidate indices list is not empty, select the index corresponding to the minimum public IO execution time
        if candidate_indices_list:
            extreme_index = min(candidate_indices_list, key=lambda idx: public_io_time_double_list[idx][pie_index])
        else:
            # If there are no candidate values (i.e., no public IO pass rate greater than 0.99), default to 0 or use other strategies as needed
            # print(f"### No candidate values (i.e., no public IO pass rate greater than 0.99), using default index: {chosen_index}")
            extreme_index = 0

        # Extract the corresponding IO pass rate and execution time based on the selected index
        selected_extreme_io_pass_rate_list.append(private_io_pass_double_list[extreme_index][pie_index])
        selected_min_execution_time_list.append(private_io_time_double_list[extreme_index][pie_index])

    return selected_extreme_io_pass_rate_list, selected_min_execution_time_list

    
 

# #####################################################################################################################🔖💡✅🟨
def use_public_io_evaluate_top_5_select_3_func(public_io_pass_double_list, public_io_time_double_list, private_io_pass_double_list, private_io_time_double_list, generation_count=5, submission_count=3):
    """
    4 double lists. First, I use the public IO pass rate greater than 0.99, then I select the index with the minimum public IO execution time,
    and then collect the private IO pass rate and execution time at the corresponding index:
    """

    selected_extreme_io_pass_rate_list = []
    selected_min_execution_time_list = []
    for pie_index in range(len(private_io_pass_double_list[0])):
        # Step 1: Find candidate indices where public IO pass rate is greater than 0.99
        candidate_indices_list = [n for n in range(generation_count) if public_io_pass_double_list[n][pie_index] >= PREDICTED_CORRECT_IO_THRESHOLD]
        if candidate_indices_list == []:
            candidate_indices_list = [n for n in range(generation_count) if public_io_pass_double_list[n][pie_index] >= 0.05]

        # Step 2: If there are candidate indices, select the top three with minimum execution time
        if candidate_indices_list:
            # Sort by public IO execution time, select top three indices
            sorted_candidate_indices = sorted(candidate_indices_list, key=lambda n: public_io_time_double_list[n][pie_index])
            min_time_top_three_indices = sorted_candidate_indices[:submission_count]  # Take top three indices with min time
        else:
            # If no candidate indices, use default values and output warning
            # print(f"### No candidate values (i.e., no public IO pass rate greater than 0.99), using default index: {min_time_top_three_indices}")
            min_time_top_three_indices = [0]  # Default to top three indices as 0


        # Step 3: Find candidate indices where private IO pass rate is greater than 0.99
        private_io_candidate_indices_list = [n for n in min_time_top_three_indices if private_io_pass_double_list[n][pie_index] >= PREDICTED_CORRECT_IO_THRESHOLD]
        if private_io_candidate_indices_list == []:
            private_io_candidate_indices_list = [n for n in min_time_top_three_indices if private_io_pass_double_list[n][pie_index] >= 0.05]
        if private_io_candidate_indices_list == []:
            private_io_candidate_indices_list = [n for n in min_time_top_three_indices if private_io_pass_double_list[n][pie_index] >= 0.0001]

        if private_io_candidate_indices_list:
            # Based on the selected top three indices, extract the corresponding private IO pass rate and execution time
            extreme_index = min(private_io_candidate_indices_list, key=lambda n: private_io_time_double_list[n][pie_index])
        else:
            extreme_index = 0 


        # Extract the corresponding IO pass rate and execution time based on the selected index
        selected_extreme_io_pass_rate_list.append(private_io_pass_double_list[extreme_index][pie_index])
        selected_min_execution_time_list.append(private_io_time_double_list[extreme_index][pie_index])

    return selected_extreme_io_pass_rate_list, selected_min_execution_time_list






# #####################################################################################################################🔖💡✅🟨
def use_private_io_evaluate_top_generate_n_submit_n_func(generation_count, submission_count, private_io_pass_double_list, private_io_time_double_list):

    selected_extreme_io_pass_rate_list = []
    selected_min_execution_time_list = []
    for pie_index in range(len(private_io_pass_double_list[0])):
        # Step 1: Find candidate indices where IO pass rate is greater than 0.99
        candidate_indices_list = [n for n in range(generation_count) if private_io_pass_double_list[n][pie_index] >= PREDICTED_CORRECT_IO_THRESHOLD]
        if candidate_indices_list == []:
            candidate_indices_list = [n for n in range(generation_count) if private_io_pass_double_list[n][pie_index] >= 0.05]
        if candidate_indices_list == []:
            candidate_indices_list = [n for n in range(generation_count) if private_io_pass_double_list[n][pie_index] >= 0.0001]

        # Step 2: If candidate indices list is not empty, select the index corresponding to the minimum IO execution time
        if candidate_indices_list:
            extreme_index = min(candidate_indices_list, key=lambda n: private_io_time_double_list[n][pie_index])
        else:
            # If there are no candidate values (i.e., no public IO pass rate greater than 0.99), default to 0 or use other strategies as needed
            extreme_index = 0

        # Extract the corresponding IO pass rate and execution time based on the selected index
        selected_extreme_io_pass_rate_list.append(private_io_pass_double_list[extreme_index][pie_index])
        selected_min_execution_time_list.append(private_io_time_double_list[extreme_index][pie_index])

    return selected_extreme_io_pass_rate_list, selected_min_execution_time_list
    





# #####################################################################################################################🔖💡✅🟨
def average_io_accuracy_func(lst, threshold=PREDICTED_CORRECT_IO_THRESHOLD):
    assert 1 >= lst[0] >= 0, "### IO Pass Rate not between 0-1! "
    passed_io_count = sum(1 for x in lst if x >= threshold)
    pass_rate = passed_io_count / len(lst)
    pass_rate = round(pass_rate * 100, 2)  # Convert to percentage
    return pass_rate




# #####################################################################################################################🔖💡✅🟨
def output_optimization_metrics(slow_io_list, fast_io_list, slow_time_list, fast_time_list):
    assert len(slow_io_list) == len(fast_io_list) == len(slow_time_list) == len(fast_time_list), "### Length mismatch between slow list and fast list! "
    error_dict = {
            "No IO Passed": 0, 
            "Partial IO Passed": 0, 
            "Slower than Slow Code": 0, 
            "Good Code": 0, 
        }
    time_optimization_rate_list = []
    speedup_ratio_list = []
    

    for i in tqdm(range(len(slow_time_list))):  # Iterate through each row
        if slow_io_list[i] <= STANDARD_CORRECT_IO_THRESHOLD:
            print(f"### Error: slow_io_list[i] <= STANDARD_CORRECT_IO_THRESHOLD")
            pass
        elif slow_time_list[i] > 12345678:
            print(f"### Error: slow_io_list[i] <= STANDARD_CORRECT_IO_THRESHOLD")
            pass
        elif fast_io_list[i] == 0:
            error_dict["No IO Passed"] += 1 
            time_optimization_rate_list.append(0)
            speedup_ratio_list.append(1)
        elif fast_io_list[i] <= PREDICTED_CORRECT_IO_THRESHOLD:
            error_dict["Partial IO Passed"] += 1 
            time_optimization_rate_list.append(0)
            speedup_ratio_list.append(1)
        elif fast_time_list[i] > 12345678:
            time_optimization_rate_list.append(0)
            speedup_ratio_list.append(1)
        else:
            slow_time = float(slow_time_list[i])
            fast_time = float(fast_time_list[i])
            if fast_time < slow_time:
                error_dict["Good Code"] += 1 
                time_optimization_rate = round(((slow_time - fast_time) / fast_time)*100, 2)
                speedup_ratio = round(slow_time / fast_time, 2)
                assert time_optimization_rate >= 0, "### Time Optimization Rate less than 0! "
                assert speedup_ratio >= 1, "### Speedup Ratio less than 1! "
                time_optimization_rate_list.append(time_optimization_rate)
                speedup_ratio_list.append(speedup_ratio)
            else:
                error_dict["Slower than Slow Code"] += 1 
                time_optimization_rate_list.append(0)
                speedup_ratio_list.append(1)

    # ---------------------------------------------------------------------------------------
    overall_optimization_percentage = round(sum(1 for x in time_optimization_rate_list if x > 10)/(len(time_optimization_rate_list))*100, 2)
    overall_speedup_ratio = round(statistics.mean(speedup_ratio_list)*100, 2)

    assert len(time_optimization_rate_list) == len(speedup_ratio_list), "### Length mismatch between Overall Optimization Percentage and Overall Speedup Ratio! "
    assert error_dict["No IO Passed"] + error_dict["Partial IO Passed"] + error_dict["Slower than Slow Code"] + error_dict["Good Code"] == len(time_optimization_rate_list), "### Length mismatch between Error Dictionary and Time Optimization Rate List! "

    print(f"### Total Evaluated: {len(time_optimization_rate_list)}")

    print(f"### Overall Optimization Percentage: {overall_optimization_percentage}")
    print(f"### Overall Speedup Ratio: {overall_speedup_ratio}")

    print(f"### Analysis Error Dictionary: {error_dict}")

    # ---------------------------------------------------------
    if DEBUG:
        print(f"\n💡💡💡 Speedup Ratio List: {speedup_ratio_list}")
        print("Indices where Speedup Ratio exceeds 200: ", [i for i, v in enumerate(speedup_ratio_list) if v > 200])

    return overall_optimization_percentage, overall_speedup_ratio




# #####################################################################################################################🔖💡✅🟨
def analyze_code_error_types_func(evaluation_result_list):

    error_type_list = []
    for evaluation_result_str in evaluation_result_list:
        evaluation_result_dict = eval(evaluation_result_str)
        single_error_type = evaluation_result_dict['Error_Type']
        if isinstance(single_error_type, list):
            error_type_list.extend(single_error_type)
        else:
            error_type_list.append(single_error_type)

    # Use Counter method
    counter_counts = Counter(error_type_list)
    # Only keep count >= 2
    # filtered = {err: cnt for err, cnt in counter_counts.items() if cnt >= 2}
    print("### Using Counter Error Type List Statistics Result: ",)
    pprint.pprint(counter_counts)
        




# #####################################################################################################################🔖💡✅🟨
def statistics_generate_io_filtering_top_k_func(Top_k):
    """
    4 double lists. First, I use the public IO pass rate greater than 0.99, then I select the index with the minimum public IO execution time,
    and then collect the private IO pass rate and execution time at the corresponding index:
    """
    # Read CSV file  
    df = pd.read_csv(f'{DATASET_PATH}')
    
    public_io_pass_double_list = [[], [], [], [], []]
    public_io_time_double_list = [[], [], [], [], []]
    generated_io_pass_double_list = [[], [], [], [], []]
    generated_io_time_double_list = [[], [], [], [], []]
    private_io_pass_double_list = [[], [], [], [], []]
    private_io_time_double_list = [[], [], [], [], []]
    for code_idx in range(5):
        public_io_pass_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__Public_IO_pass_rate_(%)'].tolist()
        public_io_time_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__Public_time(ms)'].tolist()
        generated_io_pass_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__Gen_IO_pass_rate_(%)'].tolist()
        generated_io_time_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__Gen_time(ms)'].tolist()
        private_io_pass_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__IO_pass_rate_(%)'].tolist()
        private_io_time_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__time(ms)'].tolist()

    assert type(public_io_pass_double_list[0][0]) == type(public_io_time_double_list[0][0]) == type(generated_io_pass_double_list[0][0]) == type(generated_io_time_double_list[0][0]) == type(private_io_pass_double_list[0][0]) == type(private_io_time_double_list[0][0]) == float, "### Not float type! "

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    if Top_k == 1:
        selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_public_and_generated_io_evaluate_top_5_select_1_func(public_io_pass_double_list, public_io_time_double_list, generated_io_pass_double_list, generated_io_time_double_list, private_io_pass_double_list, private_io_time_double_list)
    elif Top_k == 3:
        selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_public_and_generated_io_evaluate_top_5_select_3_func(public_io_pass_double_list, public_io_time_double_list, generated_io_pass_double_list, generated_io_time_double_list, private_io_pass_double_list, private_io_time_double_list)
    elif Top_k == 5:
        selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_private_io_evaluate_top_generate_n_submit_n_func(private_io_pass_double_list, private_io_time_double_list)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    original_slow_code_io_list = df[f'{SLOW_CODE_NAME}__IO_pass_rate_(%)'].tolist()
    original_slow_code_time_list = df[f'{SLOW_CODE_NAME}__time(ms)'].tolist()
    assert len(original_slow_code_io_list) == len(selected_extreme_io_pass_rate_list) == len(original_slow_code_time_list) == len(selected_min_execution_time_list), "### Length mismatch! "
    assert average_io_accuracy_func(original_slow_code_io_list) == 100, "### Original slow code IO pass rate is not 100! "


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    average_io_accuracy = average_io_accuracy_func(selected_extreme_io_pass_rate_list)
    print(f"\n### {COLUMN_PREFIX}, 5 @ {Top_k}, Average IO Pass Rate: {average_io_accuracy}")
    print(f"### Original Slow Code Time List, {COLUMN_PREFIX}_list, 5 @ {Top_k}:")
    optimization_ratio, speedup_ratio = output_optimization_metrics(original_slow_code_io_list, selected_extreme_io_pass_rate_list, original_slow_code_time_list, selected_min_execution_time_list)

    return average_io_accuracy, optimization_ratio, speedup_ratio





# ############################################################################################################################################################🔖💡✅🟨
def use_public_and_generated_io_evaluate_top_5_select_1_func(public_io_pass_double_list, public_io_time_double_list, generated_io_pass_double_list, generated_io_time_double_list, private_io_pass_double_list, private_io_time_double_list):
    """
    4 double lists. First, I use the public IO pass rate greater than 0.99, then I select the index with the minimum public IO execution time,
    and then collect the private IO pass rate and execution time at the corresponding index:
    """
    selected_extreme_io_pass_rate_list = []
    selected_min_execution_time_list = []
    for pie_index in range(len(private_io_pass_double_list[0])):
        # Step 1: Find candidate indices where public IO pass rate is greater than 0.99
        # core_candidate_list = [n for n in range(5) if public_io_pass_double_list[n][pie_index] >= PREDICTED_CORRECT_IO_THRESHOLD and generated_io_pass_double_list[n][pie_index] >= PREDICTED_CORRECT_IO_THRESHOLD]
        core_candidate_list = []  # Performance worse, deprecated
        all_candidate_list = [n for n in range(5) if public_io_pass_double_list[n][pie_index] >= PREDICTED_CORRECT_IO_THRESHOLD]
        if all_candidate_list == []:
            all_candidate_list = [n for n in range(5) if public_io_pass_double_list[n][pie_index] >= 0.05]

        # Step 2: If there are candidate indices, select the one with the minimum execution time
        if core_candidate_list:
            extreme_index = min(core_candidate_list, key=lambda n: (generated_io_time_double_list[n][pie_index]))
        elif all_candidate_list:
            extreme_index = min(all_candidate_list, key=lambda n: (generated_io_time_double_list[n][pie_index]))
        else:
            # If no candidate indices, use default value
            extreme_index = 0 
        
        # Extract the corresponding IO pass rate and execution time based on the selected index
        selected_extreme_io_pass_rate_list.append(private_io_pass_double_list[extreme_index][pie_index])
        selected_min_execution_time_list.append(private_io_time_double_list[extreme_index][pie_index])

    return selected_extreme_io_pass_rate_list, selected_min_execution_time_list


    
 

# ###########################################################################################################################################################################🔖💡✅🟨
def use_public_and_generated_io_evaluate_top_5_select_3_func(public_io_pass_double_list, public_io_time_double_list, generated_io_pass_double_list, generated_io_time_double_list, private_io_pass_double_list, private_io_time_double_list):
    """
    4 double lists. First, I use the public IO pass rate greater than 0.99, then I select the index with the minimum public IO execution time,
    and then collect the private IO pass rate and execution time at the corresponding index:
    """
    selected_extreme_io_pass_rate_list = []
    selected_min_execution_time_list = []
    for pie_index in range(len(private_io_pass_double_list[0])):
        # Step 1: Find candidate indices where public IO pass rate is greater than 0.99
        # core_candidate_list = [n for n in range(5) if public_io_pass_double_list[n][pie_index] >= PREDICTED_CORRECT_IO_THRESHOLD and generated_io_pass_double_list[n][pie_index] >= PREDICTED_CORRECT_IO_THRESHOLD]
        core_candidate_list = []  # Performance worse, deprecated
        all_candidate_list = [n for n in range(5) if public_io_pass_double_list[n][pie_index] >= PREDICTED_CORRECT_IO_THRESHOLD]
        if all_candidate_list == []:
            all_candidate_list = [n for n in range(5) if public_io_pass_double_list[n][pie_index] >= 0.05]

        # Step 2: If there are candidate indices, select the top three with minimum execution time
        if core_candidate_list:
            sorted_candidate_indices = sorted(core_candidate_list, key=lambda n: (generated_io_time_double_list[n][pie_index]))
            min_time_top_three_indices = sorted_candidate_indices[:3]  # Take top three indices with min time
        elif all_candidate_list:
            # Sort by public IO execution time, select top three indices
            sorted_candidate_indices = sorted(all_candidate_list, key=lambda n: (generated_io_time_double_list[n][pie_index]))
            min_time_top_three_indices = sorted_candidate_indices[:3]  # Take top three indices with min time
        else:
            # If no candidate indices, use default values and output warning
            min_time_top_three_indices = [0]

        # Step 3: Find candidate indices where private IO pass rate is greater than 0.99
        private_io_candidate_indices_list = [n for n in min_time_top_three_indices if private_io_pass_double_list[n][pie_index] >= PREDICTED_CORRECT_IO_THRESHOLD]
        if private_io_candidate_indices_list == []:
            private_io_candidate_indices_list = [n for n in min_time_top_three_indices if private_io_pass_double_list[n][pie_index] >= 0.05]
        if private_io_candidate_indices_list == []:
            private_io_candidate_indices_list = [n for n in min_time_top_three_indices if private_io_pass_double_list[n][pie_index] >= 0.0001]

        if private_io_candidate_indices_list:
            # Based on the selected top three indices, extract the corresponding private IO pass rate and execution time
            extreme_index = min(private_io_candidate_indices_list, key=lambda n: private_io_time_double_list[n][pie_index])
        else:
            extreme_index = 0 

        # Extract the corresponding IO pass rate and execution time based on the selected index
        selected_extreme_io_pass_rate_list.append(private_io_pass_double_list[extreme_index][pie_index])
        selected_min_execution_time_list.append(private_io_time_double_list[extreme_index][pie_index])

    return selected_extreme_io_pass_rate_list, selected_min_execution_time_list










# #####################################################################################################################🔖💡✅🟨
def statistics_generate_n_top_k_probability_filtering_func(Top_k):

    """
    4 double lists. First, I use the public IO pass rate greater than 0.99, then I select the index with the minimum public IO execution time,
    and then collect the private IO pass rate and execution time at the corresponding index:
    """
    # Read CSV file  
    df = pd.read_csv(f'{DATASET_PATH}')
    
    public_io_pass_double_list = [[], [], [], [], []]
    public_io_time_double_list = [[], [], [], [], []]
    generated_io_pass_double_list = [[], [], [], [], []]
    generated_io_time_double_list = [[], [], [], [], []]
    private_io_pass_double_list = [[], [], [], [], []]
    private_io_time_double_list = [[], [], [], [], []]
    for code_idx in range(5):
        public_io_pass_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__Public_IO_pass_rate_(%)'].tolist()
        public_io_time_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__Public_time(ms)'].tolist()
        generated_io_pass_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__Gen_IO_pass_rate_(%)'].tolist()
        generated_io_time_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__Gen_time(ms)'].tolist()
        private_io_pass_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__IO_pass_rate_(%)'].tolist()
        private_io_time_double_list[code_idx]     = df[f'{COLUMN_PREFIX}__Predict_Fast_code_{code_idx+1}__time(ms)'].tolist()
    
    assert type(public_io_pass_double_list[0][0]) == type(public_io_time_double_list[0][0]) == type(generated_io_pass_double_list[0][0]) == type(generated_io_time_double_list[0][0]) == type(private_io_pass_double_list[0][0]) == type(private_io_time_double_list[0][0]) == float, "### Not float type! "

    # if Top_k == 1:
    #     selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_max_probability_evaluate_top_5_select_1_func(private_io_pass_double_list, private_io_time_double_list, average_log_probability_double_list)
    # elif Top_k == 3:
    #     selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_max_probability_evaluate_top_5_select_3_func(private_io_pass_double_list, private_io_time_double_list, average_log_probability_double_list)

    if Top_k == 1:
        selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_public_and_generated_io_evaluate_top_5_select_1_func(public_io_pass_rate_double_list, public_io_execution_time_double_list, generated_io_pass_rate_double_list, generated_io_execution_time_double_list, private_io_pass_rate_double_list, private_io_execution_time_double_list)
    elif Top_k == 3:
        selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_public_and_generated_io_evaluate_top_5_select_3_func(public_io_pass_rate_double_list, public_io_execution_time_double_list, generated_io_pass_rate_double_list, generated_io_execution_time_double_list, private_io_pass_rate_double_list, private_io_execution_time_double_list)
    elif Top_k == 5:
        selected_extreme_io_pass_rate_list, selected_min_execution_time_list = use_private_io_evaluate_top_generate_n_submit_n_func(private_io_pass_rate_double_list, private_io_execution_time_double_list)
    

    # ------------------------------------------------------------------------------------------------------------------------
    original_slow_code_io_list = df['input__IO_pass_rate_(%)'].tolist()
    original_slow_code_time_list = df['input__time(ms)'].tolist()
    assert len(original_slow_code_io_list) == len(selected_extreme_io_pass_rate_list) == len(original_slow_code_time_list) == len(selected_min_execution_time_list), "### Length mismatch! "
    print(f"\n\n### Original Slow Code IO List, Average IO Pass Rate: {average_io_accuracy_func(original_slow_code_io_list)}")


    # -------------------------------------------------------------------------------------------------------------------------
    print(f"\n\n### {COLUMN_PREFIX}_list, 5 @ {Top_k} _Probability Filtering, Average IO Pass Rate: {average_io_accuracy_func(selected_extreme_io_pass_rate_list)}")
    print(f"\n\n### Original Slow Code Time List, {COLUMN_PREFIX}_list, 5 @ {Top_k} _Probability Filtering:")
    output_optimization_metrics(original_slow_code_io_list, selected_extreme_io_pass_rate_list, original_slow_code_time_list, selected_min_execution_time_list)




# #####################################################################################################################🔖💡✅🟨
def use_max_probability_evaluate_top_5_select_1_func(private_io_pass_rate_double_list, private_io_execution_time_double_list, average_log_probability_double_list):

    selected_max_io_pass_rate_list = []
    selected_max_execution_time_list = []
    for problem_index in range(len(private_io_pass_rate_double_list[0])):
        # Initialize: Assume the 0th index corresponds to the maximum log probability sum
        max_index = 0
        max_log = average_log_probability_double_list[0][problem_index]
        
        # Compare log probability sums of other code strips to find the max value and its index
        for i in range(1, 5):
            if average_log_probability_double_list[i][problem_index] > max_log:
                max_log = average_log_probability_double_list[i][problem_index]
                max_index = i

        # Extract the corresponding IO pass rate and execution time based on the selected index
        selected_max_io_pass_rate_list.append(private_io_pass_rate_double_list[max_index][problem_index])
        selected_max_execution_time_list.append(private_io_execution_time_double_list[max_index][problem_index])

    return selected_max_io_pass_rate_list, selected_max_execution_time_list






# #####################################################################################################################🔖💡✅🟨
def use_max_probability_evaluate_top_5_select_3_func(private_io_pass_rate_double_list, private_io_execution_time_double_list, average_log_probability_double_list):

    selected_max_io_pass_rate_list = []
    selected_max_execution_time_list = []
    for problem_index in range(len(private_io_pass_rate_double_list[0])):

        # Sort by average log probability, select top three indices
        sorted_candidate_indices = sorted([0, 1, 2, 3, 4], key=lambda idx: average_log_probability_double_list[idx][problem_index])
        max_avg_log_prob_top_three_indices = sorted_candidate_indices[2:] 

        # Find candidate indices where private IO pass rate is greater than 0.99
        private_io_candidate_indices_list = [idx for idx in max_avg_log_prob_top_three_indices if private_io_pass_rate_double_list[idx][problem_index] >= PREDICTED_CORRECT_IO_THRESHOLD]
        if private_io_candidate_indices_list == []:
            private_io_candidate_indices_list = [idx for idx in max_avg_log_prob_top_three_indices if private_io_pass_rate_double_list[idx][problem_index] >= 0.05]


        if private_io_candidate_indices_list:
            # Based on the selected top three indices, extract the corresponding private IO pass rate and execution time
            private_io_sorted_candidate_indices     = sorted(private_io_candidate_indices_list, key=lambda idx: private_io_execution_time_double_list[idx][problem_index])
            extreme_index   = private_io_sorted_candidate_indices[0]
        else:
            extreme_index = 0 


        # Extract the corresponding IO pass rate and execution time based on the selected index
        selected_max_io_pass_rate_list.append(private_io_pass_rate_double_list[extreme_index][problem_index])
        selected_max_execution_time_list.append(private_io_execution_time_double_list[extreme_index][problem_index])

    return selected_max_io_pass_rate_list, selected_max_execution_time_list







# #################################################################################################################################################
if __name__ == '__main__':
    main()