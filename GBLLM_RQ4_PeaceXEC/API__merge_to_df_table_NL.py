# -*- coding: utf-8 -*-
# print(f":{}")
# #####################################################################################################################🔖💡✅🟨


import json
import pandas as pd

import os

from tqdm import tqdm
from typing import List, Tuple, Any


# #####################################################################################################################🔖💡✅🟨
DEBUG = False

DATASET_PATH = r""
GENERATED_PATH = r""
SAVE_TABLE_PATH = r""
COLUMN_PREFIX = r""


"""   Change column names   """

OVERALL_DESCRIPTION = True

# #####################################################################################################################🔖💡✅🟨
def main():
    merge_generated_io_descriptions_func(DATASET_PATH, GENERATED_PATH, SAVE_TABLE_PATH, COLUMN_PREFIX)






# #####################################################################################################################🔖💡✅🟨
def merge_generated_io_descriptions_func(dataset_path, generated_path, save_table_path, column_prefix):
    df = pd.read_csv(dataset_path)

    original_io_text_list = []
    extracted_io_dict_list = []
    all_log_probs_list = []
    avg_log_probs_list = []
    
    for problem_index in tqdm(range(len(df))):
        problem_index_str = str(problem_index).zfill(4)
        with open(f"{generated_path}/{problem_index_str}_0.txt", 'r', encoding='utf-8') as f:
            code_str = f.read().strip()
        original_io_text_list.append(code_str)

        extracted_io_dict_list.append(nl_post_processing_cot_func(code_str, problem_index))

        # if 'GPT' in generated_path:
        #     with open(f"{generated_path}/{problem_index_str}_0_log_probs.txt", 'r', encoding='utf-8') as f:
        #         log_probs = f.read().strip()
        #     log_probs_list.append(log_probs)
        if os.path.exists(f"{generated_path}/0000_0_avg_log_probs.txt"):
            with open(f"{generated_path}/{problem_index_str}_0_avg_log_probs.txt", 'r', encoding='utf-8') as f:
                avg_log_probs = f.read().strip()
            avg_log_probs_list.append(avg_log_probs)


    # ------------------------------------------------------------------------------------------
    df[f'{column_prefix}_G1__Ori'] = original_io_text_list
    df[f'{column_prefix}_G1'] = extracted_io_dict_list
    # if 'GPT' in generated_path:
    #     df[f'{column_prefix}__Log_probs'] = 
    if avg_log_probs_list:
        df[f'{column_prefix}_G1__avg_log_probs'] = avg_log_probs_list

    # Save the modified data to a new CSV file
    df.to_csv(f'{save_table_path}', index=False, encoding="utf-8")





# #####################################################################################################################🔖💡✅🟨
def io_description_post_processing_cot_func(original_io_desc_str, index, public_io_str):
    total_io_desc_str = original_io_desc_str.strip()


    # -----------------------------------------------------------------------
    if total_io_desc_str.startswith('```Description'):
        total_io_desc_str = total_io_desc_str[14:].strip()
    else:
        print(f"### total_io_desc_str, does not start with ```Description: {total_io_desc_str}")


    public_io_dict = eval(public_io_str)
    public_io_dict['description'] = []


    missing_dict = {}

    for i in range(len(public_io_dict['inputs'])):
        if total_io_desc_str.count(f'## Example {i+1}:') != 1:
            print(f"### Index: {index}")
            print(f"### Public IO Dict: {public_io_dict['inputs']}")
            print(f"### IO Unit Test Dict: {total_io_desc_str}")
            if index not in missing_dict:
                missing_dict[index] = {i}  
            else:
                missing_dict[index].add(i)

        else:
            current_desc = total_io_desc_str.split(f'## Example {i+1}:')[1].split('## Example')[0].strip()
            if current_desc.endswith('```'):
                current_desc = current_desc[:-3].strip()
            if '<description of example' in current_desc:
                print(f"\n\n======== ### Current Description: {current_desc}")


            if len(current_desc) < 50:
                print(f"### Index: {index}")
                print(f"### Public IO Dict: {public_io_dict['inputs']}")
                print(f"### IO Unit Test Dict: {total_io_desc_str}")
                

            public_io_dict['description'].append(current_desc)
            

    return public_io_dict



# #####################################################################################################################🔖💡✅🟨
def nl_post_processing_cot_func(original_io_desc_str, index):
    original_code_str = original_io_desc_str.strip()
    code_str = original_code_str
    # print(f"\n\n### original_code_str, Index: {index}. code_str: {code_str}")
    for prefix in ['description', 'Description', 'Code functionality description', 'code functionality description', 'Natural language description of code functions', 'natural language description of code functions', ]:
        if (f"```{prefix}" in code_str) or (f"```\n{prefix}" in code_str):
            break

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
            print(f"\n\n### Currently double quotes, check first line for errors: {index}\ncode_str: {code_str}")
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


    if code_str == "" or code_str == "nan":
        print(f"\n### code_str is empty, Index: {index}\noriginal_code_str:\n{original_code_str}")
        code_str = 'pass'

    return code_str.strip()




# #####################################################################################################################🔖💡✅🟨
def io_overall_description_post_processing_cot_func(original_io_desc_str, index, public_io_str):
    total_io_desc_str = original_io_desc_str.strip()
    # print(f"\n\n### Index: {index}")
    # print(f"### total_io_desc_str: {total_io_desc_str}")


    # -------------------------------------------------------------------------------------------------
    if total_io_desc_str.count('```') > 0:
        if '```Description' in total_io_desc_str:
            total_io_desc_str = total_io_desc_str.split('```Description')[1].strip().split('```')[0].strip()
        elif '```description' in total_io_desc_str:
            total_io_desc_str = total_io_desc_str.split('```description')[1].strip().split('```')[0].strip()
        elif total_io_desc_str.count('```') > 0:
            total_io_desc_str = total_io_desc_str.split('```')[1].strip().split('```')[0].strip()
        else:
            print(f"### Error, unexpected situation. total_io_desc_str, incorrect start: \n{total_io_desc_str}")
            total_io_desc_str = total_io_desc_str[14:].strip()

    total_io_desc_str = total_io_desc_str.strip().replace('\n\n', ' ').replace('\n\n', ' ')

    public_io_dict = eval(public_io_str)
    public_io_dict['overview_description'] = total_io_desc_str


    return public_io_dict













# #####################################################################################################################🔖💡✅🟨
if __name__ == "__main__":
    main()