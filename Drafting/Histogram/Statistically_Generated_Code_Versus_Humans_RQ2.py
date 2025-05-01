import sys
import re
import os
from tqdm import tqdm
import io
import tokenize
import keyword
import pandas as pd
import gc
import statistics
import pprint
import statistics


DeBug = False
dataset_path = r"PIE_ChatGPT.csv"
column_name_prefix = "Use_NL_T0_01_IO"




def main():
    # Evaluate Top_1_Function()
    # 
    original_slow_code_IO_list, original_slow_code_time_list, human_IO_pass_rate_list, human_IO_execution_time_list, selected_IO_pass_rate_list, selected_execution_time_list = statistic_top_function()
    total_evaluations, failed_count, slow_code_count, fast_code_count, faster_than_human_code_count = output_optimization_metrics(original_slow_code_IO_list, original_slow_code_time_list, human_IO_pass_rate_list, human_IO_execution_time_list, selected_IO_pass_rate_list, selected_execution_time_list)

    print(f"### Total Evaluations: {total_evaluations}")
    print(f"### Failures: {failed_count}                    ### Failure Rate: {failed_count/total_evaluations * 100} ")
    print(f"### Slow Code Count: {slow_code_count}                ### Slow Code Rate: {slow_code_count/total_evaluations * 100} ")
    print(f"### Fast Code Count: {fast_code_count}                ### Fast Code Rate: {fast_code_count/total_evaluations * 100} ")
    print(f"### Faster Than Human Code Count: {faster_than_human_code_count}    ### Faster Than Human Code Rate: {faster_than_human_code_count/total_evaluations * 100} ")


def statistic_top_function():
    """
    4 double lists. First, I select public IO pass rates greater than 0.999, then I choose the index with the smallest public IO execution time,
    and then collect the corresponding private IO pass rate and execution time at that index:
    """
    # Read CSV file  
    df = pd.read_csv(f'{dataset_path}')

    original_slow_code_IO_list = df['input__IO_pass_rate_(%)'].tolist()
    original_slow_code_time_list = df['input__time(ms)'].tolist()

    human_IO_pass_rate_list = df[f'target__IO_pass_rate_(%)'].tolist()
    human_IO_execution_time_list = df[f'target__time(ms)'].tolist()

    public_IO_pass_rate_double_list = [[], [], [], [], []]
    public_IO_execution_time_double_list = [[], [], [], [], []]
    private_IO_pass_rate_double_list = [[], [], [], [], []]
    private_IO_execution_time_double_list = [[], [], [], [], []]
    average_log_probability_double_list = [[], [], [], [], []]
    
    for code_index in range(5):
        public_IO_pass_rate_double_list[code_index] = df[f'{column_name_prefix}__Predict_Fast_code_{code_index+1}__Public_IO_pass_rate_(%)'].tolist()
        public_IO_execution_time_double_list[code_index] = df[f'{column_name_prefix}__Predict_Fast_code_{code_index+1}__Public_IO_time(ms)'].tolist()
        private_IO_pass_rate_double_list[code_index] = df[f'{column_name_prefix}__Predict_Fast_code_{code_index+1}__IO_pass_rate_(%)'].tolist()
        private_IO_execution_time_double_list[code_index] = df[f'{column_name_prefix}__Predict_Fast_code_{code_index+1}__time(ms)'].tolist()
    
    assert type(public_IO_pass_rate_double_list[0][0]) == type(public_IO_execution_time_double_list[0][0]) == type(private_IO_pass_rate_double_list[0][0]) == type(private_IO_execution_time_double_list[0][0]) == float, "### Not a float type!"

    selected_IO_pass_rate_list, selected_execution_time_list = use_public_IO_evaluate_top_5_select_1_function(public_IO_pass_rate_double_list, public_IO_execution_time_double_list, private_IO_pass_rate_double_list, private_IO_execution_time_double_list)

    return original_slow_code_IO_list, original_slow_code_time_list, human_IO_pass_rate_list, human_IO_execution_time_list, selected_IO_pass_rate_list, selected_execution_time_list


def use_public_IO_evaluate_top_5_select_1_function(public_IO_pass_rate_double_list, public_IO_execution_time_double_list, private_IO_pass_rate_double_list, private_IO_execution_time_double_list):
    """
    4 double lists. First, I select public IO pass rates greater than 0.999, then I choose the index with the smallest public IO execution time,
    and then collect the corresponding private IO pass rate and execution time at that index:
    """

    selected_IO_pass_rate_list = []
    selected_execution_time_list = []
    for problem_index in range(len(private_IO_pass_rate_double_list[0])):
        # Step 1: First, find candidate indices in public IO pass rate greater than 0.99
        candidate_index_list = []
        for idx in range(5):
            if public_IO_pass_rate_double_list[idx][problem_index] > 0.999:
                candidate_index_list.append(idx)
        
        # Step 2: If there are candidates, select the index with the minimum public IO execution time
        if candidate_index_list:
            selected_index = candidate_index_list[0]
            min_execution_time = public_IO_execution_time_double_list[selected_index][problem_index]
            for idx in candidate_index_list:
                if public_IO_execution_time_double_list[idx][problem_index] < min_execution_time:
                    min_execution_time = public_IO_execution_time_double_list[idx][problem_index]
                    selected_index = idx
        else:
            # If no candidates (i.e., no public IO pass rate greater than 0.999), we can default to index 0 or use another strategy
            selected_index = 0

        # Extract the corresponding private IO pass rate and execution time based on the selected index
        selected_IO_pass_rate_list.append(private_IO_pass_rate_double_list[selected_index][problem_index])
        selected_execution_time_list.append(private_IO_execution_time_double_list[selected_index][problem_index])

    return selected_IO_pass_rate_list, selected_execution_time_list


def average_IO_accuracy_function(lst, threshold=0.999):
    passed_IO_count = sum(1 for x in lst if x > threshold)
    pass_rate = passed_IO_count / len(lst)
    return pass_rate


def output_optimization_metrics(slow_IO_list, slow_time_list, human_IO_list, human_time_list, fast_IO_list, fast_time_list):
    assert len(slow_IO_list) == len(slow_time_list) == len(human_IO_list) == len(fast_IO_list) == len(human_time_list) == len(fast_time_list), "### Slow and Fast list lengths do not match!"

    total_evaluations = 0
    failed_count = 0
    slow_code_count = 0
    fast_code_count = 0
    faster_than_human_code_count = 0

    for i in tqdm(range(len(human_time_list))):  # Traverse each row
        if slow_IO_list[i] < 0.999:
            continue
        elif slow_time_list[i] >= 1234567890 - 1:
            continue
        elif human_IO_list[i] < 0.999:
            continue
        elif human_time_list[i] >= 1234567890 - 1:
            continue

        total_evaluations += 1

        if fast_IO_list[i] < 0.999:
            failed_count += 1
        elif fast_time_list[i] >= 1234567890 - 1:
            failed_count += 1
        elif ((human_time_list[i] - fast_time_list[i]) / fast_time_list[i]) > 0.1:
            faster_than_human_code_count += 1
        elif fast_time_list[i] < slow_time_list[i]:
            fast_code_count += 1
        elif fast_time_list[i] >= slow_time_list[i]:
            slow_code_count += 1
        else:
            print(f"### Error!!!")

    return total_evaluations, failed_count, slow_code_count, fast_code_count, faster_than_human_code_count


if __name__ == '__main__':
    main()
