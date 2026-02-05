# -*- coding: utf-8 -*-
# print(f":{}")
# #####################################################################################################################🔖💡✅🟨✓✗


import json
import pandas as pd
import argparse

import os
import pprint

from tqdm import tqdm
import statistics

import numpy as np
from itertools import combinations
from API__code_sanitization_Python import simple_clean_and_remove_comments_python_func
# from API__code_clean_Cpp__latest5 import simple_code_clean_cpp_func
from API__code_sanitization_Cpp import simple_clean_and_remove_comments_cpp_func


# #####################################################################################################################🔖💡✅🟨
DEBUG = False

HISTORY_EDIT_COLUMN_PREFIXES = [
    'SBLLM_cot_G5',
    'Cot_NL_CFG_SlowMidFastTime_Round1_G5',
    'Cot_NL_CFG_SlowMidFastTime_Round2_G5',
    'Cot_NL_CFG_SlowMidFastTime_Round3_G5',
    'Cot_NL_CFG_SlowMidFastTime_Round4_G5',
]


# #####################################################################################################################🔖💡✅🟨
def get_hyperparameters():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", default=r"", type=str)
    parser.add_argument("--save_set_path", default='', type=str)
    parser.add_argument("--prepare_round_number", default=2, type=int)
    parser.add_argument("--repository_level", default=False, type=bool)

    args = parser.parse_args()

    return args


# #####################################################################################################################🔖💡✅🟨
def main():
    args = get_hyperparameters()
    print(f"###=======================Start execution, preparing round number: {args.prepare_round_number}")
    print(pprint.pformat(vars(args)))
    if args.repository_level:
        sort_repository_level_code_by_time_func(args)
    else:
        sort_cot_result_code_by_time_func(args)


# #####################################################################################################################🔖💡✅🟨
def sort_cot_result_code_by_time_func(args):

    # ---------------------------------
    dataset_path = args.dataset_path
    save_set_path = args.save_set_path
    prepare_round_number = args.prepare_round_number

    count_zero_first = 0
    total_count = 0

    # Read CSV file
    df = pd.read_csv(f'{dataset_path}')
    print(f"\n### Shape: df.shape:{df.shape}")

    column_name_list = df.columns.tolist()
    print(f"\n### Column Name List:\n")
    
    slow_code_list = df['Slow_Code'].tolist()
    slow_pass_io_list = df['input__Public_IO_pass_rate_(%)'].tolist()
    slow_time_list = df['input__Public_time(ms)'].tolist()
    assert len(slow_code_list) == len(slow_pass_io_list) == len(slow_time_list), "Length mismatch!"


    sbllm_code_double_list = [slow_code_list]
    sbllm_pass_io_double_list = [slow_pass_io_list]
    sbllm_time_double_list = [slow_time_list]

    for generation_n in range(1, 6):
        sbllm_code_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[0]}__Predict_Fast_code_{generation_n}'].tolist())
        sbllm_pass_io_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[0]}__Predict_Fast_code_{generation_n}__Public_IO_pass_rate_(%)'].tolist())
        sbllm_time_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[0]}__Predict_Fast_code_{generation_n}__Public_time(ms)'].tolist())
        assert len(sbllm_code_double_list[-1]) == len(slow_code_list), "Length mismatch!"
    
    for round_num in range(1, prepare_round_number):
        for generation_n in range(1, 6):
            sbllm_code_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[round_num]}__Predict_Fast_code_{generation_n}'].tolist())
            sbllm_pass_io_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[round_num]}__Predict_Fast_code_{generation_n}__Public_IO_pass_rate_(%)'].tolist())
            sbllm_time_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[round_num]}__Predict_Fast_code_{generation_n}__Public_time(ms)'].tolist())
    
    assert len(sbllm_code_double_list) == 1 + (5 * prepare_round_number), "Length mismatch!"
    history_edit_count = len(sbllm_code_double_list)

    return_prompt_slow_code = []
    return_prompt_mid_code = []
    return_prompt_fast_code = [] 
    return_prompt_slow_time = []
    return_prompt_mid_time = []
    return_prompt_fast_time = [] 
    expected_execution_time_list = [] 
    
    for idx in tqdm(range(len(df))):
        single_row_code_list = [sbllm_code_double_list[list_round][idx] for list_round in range(history_edit_count)]
        single_row_pass_io_list = [sbllm_pass_io_double_list[list_round][idx] for list_round in range(history_edit_count)]
        single_row_time_list = [sbllm_time_double_list[list_round][idx] for list_round in range(history_edit_count)]

        # First filter indices that pass IO, then sort by time
        sorted_indices = sorted([i for i in range(len(single_row_pass_io_list)) if (single_row_pass_io_list[i] == 1) and (single_row_time_list[i] < 1234567)], key=lambda k: single_row_time_list[k], reverse=True)
        assert 0 < len(sorted_indices) <= len(single_row_code_list), "Length mismatch!"

        sorted_code_list = [single_row_code_list[i] for i in sorted_indices]
        sorted_time_list = [single_row_time_list[i] for i in sorted_indices]

        # Remove two excess elements
        if len(sorted_indices) >= 4:
            indices_to_keep_list, min_variance = find_best_index_combination_to_remove_elements_func(sorted_indices, sorted_time_list)
            sorted_code_list = [single_row_code_list[i] for i in indices_to_keep_list]
            sorted_time_list = [single_row_time_list[i] for i in indices_to_keep_list]
            # current_expected_execution_time = int(0.7 * sorted_time_list[-1])

            # ----------------------------
            if indices_to_keep_list[0] == 0:
                count_zero_first += 1
            total_count += 1

        # --------------------------------------------------------------------------
        if sorted_time_list[-1] < 5:
            _expected_execution_time = sorted_time_list[-1] * 0.7
        else:
            _expected_execution_time = int(sorted_time_list[-1] * 0.7)
            
        if len(sorted_code_list) == 1:
            return_prompt_slow_code.append(sorted_code_list[0])
            return_prompt_mid_code.append('pass')
            return_prompt_fast_code.append('pass')
            return_prompt_slow_time.append(sorted_time_list[0])
            return_prompt_mid_time.append('pass')
            return_prompt_fast_time.append('pass')
            expected_execution_time_list.append(_expected_execution_time)
        elif len(sorted_code_list) == 2:
            return_prompt_slow_code.append(sorted_code_list[0])
            return_prompt_mid_code.append(sorted_code_list[1])
            return_prompt_fast_code.append('pass')
            return_prompt_slow_time.append(sorted_time_list[0])
            return_prompt_mid_time.append(sorted_time_list[1])
            return_prompt_fast_time.append('pass')
            expected_execution_time_list.append(_expected_execution_time)
        elif len(sorted_code_list) == 3:
            return_prompt_slow_code.append(sorted_code_list[0])
            return_prompt_mid_code.append(sorted_code_list[1])
            return_prompt_fast_code.append(sorted_code_list[2])
            return_prompt_slow_time.append(sorted_time_list[0])
            return_prompt_mid_time.append(sorted_time_list[1])
            return_prompt_fast_time.append(sorted_time_list[2])
            expected_execution_time_list.append(_expected_execution_time)

    # ----------------------------------------------------------------------
    if ("_Py_" in args.dataset_path) and ("_Cpp_" not in args.dataset_path):
        print(f"\n✓✓✓ Python Simple Code Clean and Remove Comments Started...")
        return_prompt_slow_code = [simple_clean_and_remove_comments_python_func(code) for code in return_prompt_slow_code]
        return_prompt_mid_code = [simple_clean_and_remove_comments_python_func(code) for code in return_prompt_mid_code]
        return_prompt_fast_code = [simple_clean_and_remove_comments_python_func(code) for code in return_prompt_fast_code]
    elif ("_Cpp_" in args.dataset_path) and ("_Py_" not in args.dataset_path):
        print(f"\n✓✓✓ Cpp Simple Code Clean and Remove Comments Started...")
        return_prompt_slow_code = [simple_clean_and_remove_comments_cpp_func(code) for code in return_prompt_slow_code]
        return_prompt_mid_code = [simple_clean_and_remove_comments_cpp_func(code) for code in return_prompt_mid_code]
        return_prompt_fast_code = [simple_clean_and_remove_comments_cpp_func(code) for code in return_prompt_fast_code]
    else:
        raise ValueError("✗✗✗ Dataset path must contain either _Py_ or _Cpp_!")

    # ----------------------------------------------------------------------
    if prepare_round_number == 1:
        df['SBLLM_cot_Round1_Sorted_SlowCode'] = return_prompt_slow_code
        df['SBLLM_cot_Round1_Sorted_MidCode'] = return_prompt_mid_code
        df['SBLLM_cot_Round1_Sorted_FastCode'] = return_prompt_fast_code
        df['SBLLM_cot_Round1_Sorted_SlowTime'] = return_prompt_slow_time
        df['SBLLM_cot_Round1_Sorted_MidTime'] = return_prompt_mid_time
        df['SBLLM_cot_Round1_Sorted_FastTime'] = return_prompt_fast_time
        df['SBLLM_cot_Round1_ExpectedTime'] = expected_execution_time_list
    elif prepare_round_number > 1:
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_Sorted_SlowCode'] = return_prompt_slow_code
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_Sorted_MidCode'] = return_prompt_mid_code
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_Sorted_FastCode'] = return_prompt_fast_code
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_Sorted_SlowTime'] = return_prompt_slow_time
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_Sorted_MidTime'] = return_prompt_mid_time
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_Sorted_FastTime'] = return_prompt_fast_time
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_ExpectedTime'] = expected_execution_time_list


    # Automatically generate new name
    if save_set_path == '':
        prefix_path = '__'.join(dataset_path.split('__')[:-1])
        pie_number = dataset_path.split('__')[-2].split('_')[-2]
        prefix_path = prefix_path.replace(pie_number, str(int(pie_number)+1).zfill(3))
        _save_set_path = f"{prefix_path}__sorted_cot_result_code_by_time.csv"
    else:
        _save_set_path = save_set_path

    # Save the modified data to a new CSV file
    df.to_csv(f'{_save_set_path}', index=False)

    print(f"\n✅✅✅ Saved: {_save_set_path}")
    print(f"\n✅✅✅ Count zero first: {count_zero_first} / Total: {total_count} , Ratio: {count_zero_first/(total_count if total_count else 1):.2%} ")


# #####################################################################################################################🔖💡✅🟨
def find_best_index_combination_to_remove_elements_func(code_index_list, code_time_list):
    """
    Find the best combination of indices to remove elements, making the second list as evenly distributed as possible.
    
    Args:
    code_index_list: The first list (indices)
    code_time_list: The second list (times)
    
    Returns:
    indices_to_keep_list: The list of indices to keep
    min_variance: The corresponding minimum variance
    """
    
    n = len(code_index_list)
    k = n - 3  # Number of elements to remove
    
    # Constraint conditions for elements that cannot be removed
    forbidden_indices = set()
    
    # Constraint 1: Cannot remove the index of element 0 in the first list
    for i, val in enumerate(code_index_list):
        if val == 0:
            forbidden_indices.add(i)
    
    # Constraint 2: Cannot remove index 0 and the last index
    forbidden_indices.add(0)
    forbidden_indices.add(n-1)
    
    # All possible indices
    all_indices = set(range(n))
    
    # Generate all possible removal combinations
    all_possible_removal_combinations = list(combinations(all_indices, k))
    
    # print(f"List length: {n}, need to remove {k} elements")
    # print(f"Forbidden removal indices: {sorted(forbidden_indices)}")
    # print(f"Number of possible removal combinations: {len(all_possible_removal_combinations)}")
    
    best_removal_index_combination_list = None
    min_variance = float('inf')
    
    # Calculate variance after removing each candidate combination
    for current_removal_index_combination in all_possible_removal_combinations:
        # Check if it contains forbidden indices
        if any(idx in forbidden_indices for idx in current_removal_index_combination):
            continue
        
        # Create new list (remove elements at corresponding indices)
        new_code_time_list = [code_time_list[i] for i in range(n) if i not in current_removal_index_combination]
        
        # Calculate variance
        current_variance = np.var(new_code_time_list)
        
        # print(f"Remove indices {current_removal_index_combination}: List_B = {new_code_time_list}, Variance = {current_variance:.2f}")
        
        # Update best choice
        if current_variance < min_variance:
            min_variance = current_variance
            best_removal_index_combination_list = current_removal_index_combination
    
    indices_to_keep_list = [val for i, val in enumerate(code_index_list) if i not in best_removal_index_combination_list]

    return indices_to_keep_list, min_variance


# #####################################################################################################################🔖💡✅🟨
def sort_repository_level_code_by_time_func(args):

    # ---------------------------------
    dataset_path = args.dataset_path
    save_set_path = args.save_set_path
    prepare_round_number = args.prepare_round_number

    count_zero_first = 0
    total_count = 0

    # Read CSV file
    df = pd.read_csv(f'{dataset_path}')
    print(f"\n### Shape: df.shape:{df.shape}")
    
    slow_code_list = df['Slow_Code'].tolist()
    slow_pass_io_list = df['input__IO_pass_rate_(%)'].tolist()
    slow_time_list = df['input__time(us)'].tolist()
    assert len(slow_code_list) == len(slow_pass_io_list) == len(slow_time_list), "Length mismatch!"

    sbllm_code_double_list = [slow_code_list]
    sbllm_pass_io_double_list = [slow_pass_io_list]
    sbllm_time_double_list = [slow_time_list]

    for round_num in range(1, prepare_round_number):
        for generation_n in range(1, 6):
            sbllm_code_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[round_num]}__Predict_Fast_code_{generation_n}'].tolist())
            sbllm_pass_io_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[round_num]}__Predict_Fast_code_{generation_n}__IO_pass_rate_(%)'].tolist())
            sbllm_time_double_list.append(df[f'{HISTORY_EDIT_COLUMN_PREFIXES[round_num]}__Predict_Fast_code_{generation_n}__time(us)'].tolist())
    
    assert len(sbllm_code_double_list) == 1 + (5 * (prepare_round_number - 1)), "Length mismatch!"
    history_edit_count = len(sbllm_code_double_list)

    return_prompt_slow_code = []
    return_prompt_mid_code = []
    return_prompt_fast_code = [] 
    return_prompt_slow_time = []
    return_prompt_mid_time = []
    return_prompt_fast_time = [] 
    expected_execution_time_list = [] 
    
    for idx in tqdm(range(len(df))):
        single_row_code_list = [sbllm_code_double_list[list_round][idx] for list_round in range(history_edit_count)]
        single_row_pass_io_list = [sbllm_pass_io_double_list[list_round][idx] for list_round in range(history_edit_count)]
        single_row_time_list = [sbllm_time_double_list[list_round][idx] for list_round in range(history_edit_count)]

        # First filter indices that pass IO, then sort by time
        sorted_indices = sorted([i for i in range(len(single_row_pass_io_list)) if (single_row_pass_io_list[i] == 1) and (single_row_time_list[i] < 123456789)], key=lambda k: single_row_time_list[k], reverse=True)
        # if len(sorted_indices) == 0:
        #     print(f"✗✗✗ Warning: idx={idx} no code passed IO, skipping this row!")
        assert 0 < len(sorted_indices) <= len(single_row_code_list), "Length mismatch!"

        sorted_code_list = [single_row_code_list[i] for i in sorted_indices]
        sorted_time_list = [single_row_time_list[i] for i in sorted_indices]

        # Remove two excess elements
        if len(sorted_indices) >= 4:
            indices_to_keep_list, min_variance = find_best_index_combination_to_remove_elements_func(sorted_indices, sorted_time_list)
            sorted_code_list = [single_row_code_list[i] for i in indices_to_keep_list]
            sorted_time_list = [single_row_time_list[i] for i in indices_to_keep_list]
            # current_expected_execution_time = int(0.7 * sorted_time_list[-1])

            # ----------------------------
            if indices_to_keep_list[0] == 0:
                count_zero_first += 1
            total_count += 1

        # --------------------------------------------------------------------------
        if sorted_time_list[-1] < 5:
            _expected_execution_time = sorted_time_list[-1] * 0.7
        else:
            _expected_execution_time = int(sorted_time_list[-1] * 0.7)
        if len(sorted_code_list) == 1:
            return_prompt_slow_code.append(sorted_code_list[0])
            return_prompt_mid_code.append('pass')
            return_prompt_fast_code.append('pass')
            return_prompt_slow_time.append(sorted_time_list[0])
            return_prompt_mid_time.append('pass')
            return_prompt_fast_time.append('pass')
            expected_execution_time_list.append(_expected_execution_time)
        elif len(sorted_code_list) == 2:
            return_prompt_slow_code.append(sorted_code_list[0])
            return_prompt_mid_code.append(sorted_code_list[1])
            return_prompt_fast_code.append('pass')
            return_prompt_slow_time.append(sorted_time_list[0])
            return_prompt_mid_time.append(sorted_time_list[1])
            return_prompt_fast_time.append('pass')
            expected_execution_time_list.append(_expected_execution_time)
        elif len(sorted_code_list) == 3:
            return_prompt_slow_code.append(sorted_code_list[0])
            return_prompt_mid_code.append(sorted_code_list[1])
            return_prompt_fast_code.append(sorted_code_list[2])
            return_prompt_slow_time.append(sorted_time_list[0])
            return_prompt_mid_time.append(sorted_time_list[1])
            return_prompt_fast_time.append(sorted_time_list[2])
            expected_execution_time_list.append(_expected_execution_time)
    
    

    # ----------------------------------------------------------------------
    if prepare_round_number == 1:
        df['SBLLM_cot_Round1_Sorted_SlowCode'] = return_prompt_slow_code
        df['SBLLM_cot_Round1_Sorted_MidCode'] = return_prompt_mid_code
        df['SBLLM_cot_Round1_Sorted_FastCode'] = return_prompt_fast_code
        df['SBLLM_cot_Round1_Sorted_SlowTime'] = return_prompt_slow_time
        df['SBLLM_cot_Round1_Sorted_MidTime'] = return_prompt_mid_time
        df['SBLLM_cot_Round1_Sorted_FastTime'] = return_prompt_fast_time
        df['SBLLM_cot_Round1_ExpectedTime'] = expected_execution_time_list
    elif prepare_round_number > 1:
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_Sorted_SlowCode'] = return_prompt_slow_code
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_Sorted_MidCode'] = return_prompt_mid_code
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_Sorted_FastCode'] = return_prompt_fast_code
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_Sorted_SlowTime'] = return_prompt_slow_time
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_Sorted_MidTime'] = return_prompt_mid_time
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_Sorted_FastTime'] = return_prompt_fast_time
        df[f'Cot_NL_CFG_SlowMidFast_Round{prepare_round_number}_ExpectedTime'] = expected_execution_time_list


    # Automatically generate new name
    if save_set_path == '':
        prefix_path = '__'.join(dataset_path.split('__')[:-1])
        pie_number = dataset_path.split('__')[-2].split('_')[-2]
        prefix_path = prefix_path.replace(pie_number, str(int(pie_number)+1).zfill(3))
        _save_set_path = f"{prefix_path}__sorted_cot_result_code_by_time.csv"
    else:
        _save_set_path = save_set_path

    # Save the modified data to a new CSV file
    df.to_csv(f'{_save_set_path}', index=False)

    print(f"\n✅✅✅ Saved: {_save_set_path}")
    print(f"\n✅✅✅ Count zero first: {count_zero_first} / Total: {total_count} , Ratio: {count_zero_first/(total_count if total_count else 1):.2%} ")


# #####################################################################################################################🔖💡✅🟨
if __name__ == "__main__":
    print("############################################################################################################🔖💡✅🟨 Start File Exclusive__Sort_COT_Result_Code_By_Time.py")
    main()