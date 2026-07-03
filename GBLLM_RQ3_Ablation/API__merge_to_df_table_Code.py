# -*- coding: utf-8 -*-
# print(f":{}")
# #####################################################################################################################🔖💡✅🟨❌


import json
import pandas as pd

import os

from tqdm import tqdm
# Assuming the external files are named this way, but the functions inside need translation in the import
from API__code_sanitization_Python import heavy_code_sanitization_python_func
from API__code_sanitization_Cpp import simple_code_sanitization_cpp_func


# #####################################################################################################################🔖💡✅🟨❌
DEBUG = True

DATASET_PATH = r""
GENERATED_PATH = r""
SAVE_TABLE_PATH = r""
COLUMN_PREFIX = r""
"""   Change column names   """


# #####################################################################################################################🔖💡✅🟨
def main():
    merge_to_df_fast_code_func(DATASET_PATH, GENERATED_PATH, SAVE_TABLE_PATH, COLUMN_PREFIX)


# #####################################################################################################################🔖💡✅🟨
def merge_to_df_fast_code_func(dataset_path, generated_path, save_table_path, column_prefix):
    if ('_Py' in dataset_path) and ('_Cpp' not in dataset_path):
        programming_language = 'python'
    elif ('_Cpp' in dataset_path) and ('_Py' not in dataset_path):
        programming_language = 'cpp'
        
    if os.path.exists(f"{generated_path}/0000_4.txt"):
        process_n5_content_func(dataset_path, generated_path, save_table_path, column_prefix, programming_language=programming_language)        
    else:
        single_code_func(dataset_path, generated_path, save_table_path, column_prefix, programming_language=programming_language)


# #####################################################################################################################🔖💡✅🟨
def single_code_func(dataset_path, generated_path, save_table_path, column_prefix, programming_language):
    df = pd.read_csv(dataset_path)

    original_generated_code_list = []
    post_processed_code_list = []
    avg_log_probs_list = []
    
    for problem_index in tqdm(range(len(df))):
        problem_index_str = str(problem_index).zfill(4)
        with open(f"{generated_path}/{problem_index_str}_0.txt", 'r', encoding='utf-8') as f:
            code_str = f.read().strip()
        original_generated_code_list.append(code_str)
        post_processed_code_list.append(cot_code_post_processing_py_cpp_func(code_str, problem_index, programming_language).strip())

        if os.path.exists(f"{generated_path}/{problem_index_str}_0_avg_log_probs.txt"):
            with open(f"{generated_path}/{problem_index_str}_0_avg_log_probs.txt", 'r', encoding='utf-8') as f:
                avg_log_probs = f.read().strip()
            avg_log_probs_list.append(avg_log_probs)


    # ------------------------------------------------------------------------------------------
    df[f'{column_prefix}_G1__Predict_Fast_code__Ori'] = original_generated_code_list
    df[f'{column_prefix}_G1__Predict_Fast_code'] = post_processed_code_list
    if len(avg_log_probs_list) > 9:
        df[f'{column_prefix}_G1__avg_log_probs'] = avg_log_probs_list

    # Save the modified data to a new CSV file  
    df.to_csv(f'{save_table_path}', index=False, encoding="UTF-8")


# #####################################################################################################################🔖💡✅🟨
def process_n5_content_func(dataset_path, generated_path, save_table_path, column_prefix, programming_language):
    df = pd.read_csv(dataset_path)

    original_generated_code_double_list = [[], [], [], [], []]
    post_processed_code_double_list = [[], [], [], [], []]
    avg_log_probs_double_list = [[], [], [], [], []]
    
    for problem_index in tqdm(range(len(df))):
        for code_sample_index in range(5):
            problem_index_str = str(problem_index).zfill(4)
            with open(f"{generated_path}/{problem_index_str}_{code_sample_index}.txt", 'r', encoding='utf-8') as f:
                code_str = f.read().strip()
            original_generated_code_double_list[code_sample_index].append(code_str)
            post_processed_code_double_list[code_sample_index].append(cot_code_post_processing_py_cpp_func(code_str, problem_index, programming_language).strip())

            if os.path.exists(f"{generated_path}/{problem_index_str}_{code_sample_index}_avg_log_probs.txt"):
                with open(f"{generated_path}/{problem_index_str}_{code_sample_index}_avg_log_probs.txt", 'r', encoding='utf-8') as f:
                    avg_log_probs = f.read().strip()
                avg_log_probs_double_list[code_sample_index].append(avg_log_probs)


    # --------------------------------------------------------------------------------------------
    for code_sample_index in range(5):
        df[f'{column_prefix}_G5__Predict_Fast_code__Ori_{code_sample_index+1}'] = original_generated_code_double_list[code_sample_index]
        df[f'{column_prefix}_G5__Predict_Fast_code_{code_sample_index+1}']     = post_processed_code_double_list[code_sample_index]
        if len(avg_log_probs_double_list[code_sample_index]) > 9:
            df[f'{column_prefix}_G5__avg_log_probs_{code_sample_index+1}'] = avg_log_probs_double_list[code_sample_index]

    # Save the modified data to a new CSV file  
    df.to_csv(f'{save_table_path}', index=False, encoding="UTF-8")


# ```bash
# ```markdown
# ```c++


# #####################################################################################################################🔖💡✅🟨
def cot_code_post_processing_py_cpp_func(original_code_str, index, prefix='python'):
    code_str = original_code_str.strip()
    # print(f"\n\n### original_code_str, Index: {index}. code_str: {code_str}")
    for prefix_item in ['python', 'Python', 'cpp', 'Cpp', 'c++', 'C++', ' python', ' Python', ' cpp', ' Cpp', ' c++', ' C++', 'css', 'bash', 'markdown', 'perl', 'php', 'scss', 'yaml', 'sh', 'sql', 'C', 'c', 'END_ERROR_MARKER',]:
        if (f"```{prefix_item}" in code_str) or (f"```\n{prefix_item}" in code_str):
            prefix = prefix_item
            break
        if prefix_item == 'END_ERROR_MARKER':
            print(f"\n\n### Prefix Error: {prefix_item}, Index: {index}. code_str[:20]: {code_str}")


    # Handle single triple-quote cases
    if code_str.count("```") == 1:
        # print(f"\n\n### ========================== Error, Single triple quote, Index: {index}\noriginal_code_str:\n{original_code_str}")
        if f"```{prefix}" in code_str:
            code_str = code_str.split(f"```{prefix}")[1].strip()
        elif f"```\n{prefix}" in code_str:
            code_str = code_str.split(f"```\n{prefix}")[1].strip()
        elif f"```\n" in code_str:
            code_str = code_str.split(f"```\n")[1].split("```")[0].strip()
            print(f"\n\n### Single quote, check first line for errors: {index}\ncode_str[:20]: {code_str[:20]}")
        else:
            print(f"\n\n### Needs careful distinction: {index}\noriginal_code_str:\n{original_code_str}")
            code_str = code_str.split(f"```")[1].split("```")[0].strip()
        assert "```" not in code_str


    # Handle double triple-quote cases
    elif code_str.count("```") == 2:
        if f"```{prefix}" in code_str:
            code_str = code_str.split(f"```{prefix}")[1].split("```")[0].strip()
        elif f"```\n{prefix}" in code_str:
            code_str = code_str.split(f"```\n{prefix}")[1].split("```")[0].strip()
        elif f"```\n" in code_str:
            code_str = code_str.split(f"```\n")[1].split("```")[0].strip()
            print(f"\n\n### Double quotes, check first line for errors: {index}\ncode_str[:20]: {code_str[:20]}")
        else:
            print(f"\n\n### Needs careful distinction: {index}\noriginal_code_str:\n{original_code_str}")
            code_str = code_str.split(f"```")[1].split("```")[0].strip()
        assert "```" not in code_str


    # Handle multiple triple-quote cases
    elif code_str.count("```") > 2:
        # print(f"\n\n### ========================== Error, Multiple triple quotes, Index: {index}\noriginal_code_str:\n{original_code_str}")

        if f"```{prefix}" in code_str:
            code_str = code_str.split(f"```{prefix}")[1].split("```")[0].strip()
        if f"```\n{prefix}" in code_str:
            code_str = code_str.split(f"```\n{prefix}")[1].split("```")[0].strip()

        if "```" in code_str:
            code_str = code_str.split("```")[1]
            if (not code_str) or (code_str[0] != '\n'):
                print(f"\n\n### ========================== Error, Multiple triple quotes, Index: {index}\noriginal_code_str:\n{original_code_str}")
                print(f"\n\n### After processing, code_str:\n{code_str}")
            code_str = code_str.strip()


        if "```" in code_str:
            print(f"\n\n### ========================== Error, Multiple triple quotes, Unresolved, Index: {index}\noriginal_code_str:\n{original_code_str}")
            print(f"\n\n### After processing, code_str:\n{code_str}")


    # ========================================================================================================
    if prefix in ['python', 'Python']:
        code_str = heavy_code_sanitization_python_func(code_str)
    elif prefix in ['cpp', 'Cpp', 'c++', 'C++', 'css', 'C', 'c']:
        code_str = simple_code_sanitization_cpp_func(code_str)


    # --------------------------------------------------------------------------------------------------------------
    if code_str == "" or code_str == "nan":
        print(f"\n### code_str is empty, Index: {index}\noriginal_code_str:\n{original_code_str}")
        code_str = 'pass'

    return code_str.strip()


# #####################################################################################################################🔖💡✅🟨
def codellama_code_post_processing_func(original_code_str, index):
    code_str = original_code_str.strip()
 
    if len(code_str.split('```')) > 1:
        code_str = code_str.split('```')[1]
    code_str = code_str.split('### Optimized version')[-1].split('# Test')[0].split('# test')[0].split('### Code Functionality Description')[0].split('### Control Flow Graph')[0]
    
    if code_str == "" or code_str == "nan":
        print(f"\n### code_str is empty, Index: {index}\ncode_str:\n{original_code_str}")
        code_str = 'pass'
        
    return code_str


# #####################################################################################################################🔖💡✅🟨
def check_code_func(tested_code_str):
    secondary_risk_keywords = ['setrecursionlimit(', 'stack_size(', "if __name__ == '__main__':", ' is not ', ' is ', 'stdout', 'stderr',
             ", file=sys.stdout", ", file=stdout", ", output=sys.stdout", ", output=stdout", ", file=sys.stderr", ", file=stderr", ", output=sys.stderr", ", output=stderr",
             "sys.stdin.readline", "stdin.readline", "sys.stdin.buffer.readline", "stdin.buffer.readline",
             "sys.stdout.write", "stdout.write", "sys.__stdout__.write", "__stdout__.write", "sys.stderr.write", "stderr.write", "sys.__stderr__.write", "__stderr__.write",
             'return sys.stdout.flush()', 
             "IOWrapper(", "FastIO(", "StringIO(", "BytesIO(", "FastStdout(", ".close(", "stdout", "stderr", 
             'open(',
             "threading", "thread", "multiprocessing", "asyncio", "queue.Queue(", "ProcessPoolExecutor", "concurrent", "fork(", "subprocess.run("]
    
    risk_keywords_list = ['setrecursionlimit(', 'stack_size(', 
             ", file=sys.stdout", ", file=stdout", ", output=sys.stdout", ", output=stdout", ", file=sys.stderr", ", file=stderr", ", output=sys.stderr", ", output=stderr",
             "sys.stdin.readline", "stdin.readline", "sys.stdin.buffer.readline", "stdin.buffer.readline",
             "sys.stdout.write", "stdout.write", "sys.__stdout__.write", "__stdout__.write", "sys.stderr.write", "stderr.write", "sys.__stderr__.write", "__stderr__.write",
             'return sys.stdout.flush()', 
             "IOWrapper(", "FastIO(", "StringIO(", "BytesIO(", "FastStdout(", ".close(",
             "threading", "thread", "multiprocessing", "asyncio", "queue.Queue(", "ProcessPoolExecutor", "concurrent", "fork(", "subprocess.run("]

    returned_code_list = []
    returned_risk_keywords_list = []
    code_lines_slice = tested_code_str.split('\n')
    for code_line in code_lines_slice:
        for risk_keyword in risk_keywords_list:
            if risk_keyword in code_line:
                returned_risk_keywords_list.append(risk_keyword)
            else:
                returned_code_list.append(code_line)
    if len(returned_risk_keywords_list) > 0:
        returned_code = '\n'.join(returned_code_list).strip()
        return True, set(returned_risk_keywords_list), returned_code
    else:
        return False, [], ''


# #####################################################################################################################🔖💡✅🟨
if __name__ == "__main__":
    main()