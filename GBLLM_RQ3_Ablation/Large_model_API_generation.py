# -*- coding: utf-8 -*-
# print(f"### :\n{}")
# #####################################################################################################################🔖💡✅🟨✓✗

import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

import os
import argparse
from tqdm import tqdm

import json
import pprint
import pandas as pd
import time
from datetime import datetime, timedelta
import importlib
import multiprocessing
import numpy as np
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

from API__Single_Generation import CodeLlama_Deepinfra_Function
from API__Single_Generation import Gemini_Official_Function
from API__Single_Generation import ChatGPT_Function
from API__Single_Generation import DeepSeek_Function


from API__merge_to_df_table_NL import merge_generated_io_descriptions_func
from API__merge_to_df_table_Code import merge_to_df_fast_code_func

  

# #####################################################################################################################🔖💡🟨 ✅❌✓✗


# #####################################################################################################################🔖💡🟨✅❌
def get_hyperparameters():

    parser = argparse.ArgumentParser()

    parser.add_argument("--is_output_prompt", default=False, type=bool)

    parser.add_argument("--core_number", default=r'')


    # Baseline and Generated DF paths
    parser.add_argument("--baseline_df_path", default=r'', type=str)
    parser.add_argument("--generated_df_path", default=r'', type=str)
    parser.add_argument("--iteration_round", default=1, type=int)
    parser.add_argument("--num_threads", default=10, type=int)



    # 💡--------------------------------------------------------------------------------------------------------
    parser.add_argument("--prompt_template_name", default="",  type=str)

    # 💡--------------------------------------------------------------------------------------------------------
    parser.add_argument("--slow_code_column", default="Slow_Code", type=str)
    parser.add_argument("--nl_column",     default="Code_Function_Description_G1", type=str)



    # 💡--------------------------------------------------------------------------------------------------------
    parser.add_argument("--io_test_column", default="Public_IO_unit_tests__Dedup", type=str)

    # Number of generated codes = batch size * repeat times. One of them must be 1.
    parser.add_argument("--num_generated_codes", default=5, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--repeat_times", default=5, type=int)

    parser.add_argument("--temperature", default=1, type=float)

    parser.add_argument("--task_description", default="")


    args = parser.parse_args()

    return args


# #####################################################################################################################🔖💡✅🟨 ✗✗✗ ✓✓✓ 
def set_extra_parameters(args):

    # Set global core number
    core_index = args.core_number
    
    args.key_index = core_index

    # -----------------------------------------------------------------------------------------
    global num_threads
    num_threads = args.num_threads

    # -----------------------------------------------------------------------------------------
    """ First enter ablation determination """
    if ("_Ablation_Replace_CFG_" in args.prompt_template_name) and ('PIE_Py' in args.baseline_df_path):
        args.cfg_column_name = "Full__Refer_Fast_CFG_1144"
    elif ("_Ablation_Replace_CFG_" in args.prompt_template_name):
        args.cfg_column_name = "Match_Full_Fast_CFG"
    elif ('PIE_' in args.baseline_df_path) and ('PIE_Py' in args.baseline_df_path):
        args.cfg_column_name = "Refer_Fast_CFG_1144"
    else:
        args.cfg_column_name = "Match_Fast_CFG"

    # -----------------------------------------------------------------------------------------
    """ First enter ablation determination """
    if ("_Ablation_Remove_NL_" in args.prompt_template_name):
        args.save_column_prefix = f'Ablation_Remove_NL_Cot_CFG_SlowMidFastTime_Round{args.iteration_round}'
    elif ("_Ablation_Remove_IO_" in args.prompt_template_name) and ("_Generate_NL_" in args.prompt_template_name):
        args.save_column_prefix = f'Ablation_Remove_IO_Code_Function_Description'
    elif ("_Ablation_Remove_IO_" in args.prompt_template_name):
        args.save_column_prefix = f'Ablation_Remove_IO_Cot_NL_CFG_SlowMidFastTime_Round{args.iteration_round}'
    elif ("_Ablation_Remove_CFG_" in args.prompt_template_name):
        args.save_column_prefix = f'Ablation_Remove_CFG_Cot_NL_SlowMidFastTime_Round{args.iteration_round}'
    elif ("_Ablation_Replace_CFG_" in args.prompt_template_name):
        args.save_column_prefix = f'Ablation_All_CFG_Cot_NL_SlowMidFastTime_Round{args.iteration_round}'
    elif ("_Ablation_Remove_Time_" in args.prompt_template_name):
        args.save_column_prefix = f'Ablation_Remove_Time_Cot_NL_SlowMidFast_Round{args.iteration_round}'
    elif ("_Ablation_Remove_Trajectory_" in args.prompt_template_name):
        args.save_column_prefix = f'Ablation_Remove_Trajectory_Cot_NL_Round{args.iteration_round}'
    elif ("_Ablation_Remove_All_" in args.prompt_template_name):
        args.save_column_prefix = f'Ablation_Remove_All_Cot_Round{args.iteration_round}'


    elif (args.iteration_round == 0):
        assert ("Generate_NL" in args.prompt_template_name) or ("Repo_Level" in args.prompt_template_name), "When iteration_round is 0, prompt_template_name must contain 'Generate_NL'"
        args.save_column_prefix = f'Code_Function_Description'
    else:
        assert "Generate_Code" in args.prompt_template_name, "When iteration_round is not 0, prompt_template_name must contain 'Generate_Code'"
        args.save_column_prefix = f'Cot_NL_CFG_SlowMidFastTime_Round{args.iteration_round}'


    # =====================================================================================
    slow_mid_fast_columns_full_list = [
    ['SBLLM_cot_Round1_Sorted_SlowCode', 'SBLLM_cot_Round1_Sorted_MidCode', 'SBLLM_cot_Round1_Sorted_FastCode', 'SBLLM_cot_Round1_Sorted_SlowTime', 'SBLLM_cot_Round1_Sorted_MidTime', 'SBLLM_cot_Round1_Sorted_FastTime', 'SBLLM_cot_Round1_ExpectedTime',],
    ['Cot_NL_CFG_SlowMidFast_Round2_Sorted_SlowCode', 'Cot_NL_CFG_SlowMidFast_Round2_Sorted_MidCode', 'Cot_NL_CFG_SlowMidFast_Round2_Sorted_FastCode', 'Cot_NL_CFG_SlowMidFast_Round2_Sorted_SlowTime', 'Cot_NL_CFG_SlowMidFast_Round2_Sorted_MidTime', 'Cot_NL_CFG_SlowMidFast_Round2_Sorted_FastTime', 'Cot_NL_CFG_SlowMidFast_Round2_ExpectedTime',],
    ['Cot_NL_CFG_SlowMidFast_Round3_Sorted_SlowCode', 'Cot_NL_CFG_SlowMidFast_Round3_Sorted_MidCode', 'Cot_NL_CFG_SlowMidFast_Round3_Sorted_FastCode', 'Cot_NL_CFG_SlowMidFast_Round3_Sorted_SlowTime', 'Cot_NL_CFG_SlowMidFast_Round3_Sorted_MidTime', 'Cot_NL_CFG_SlowMidFast_Round3_Sorted_FastTime', 'Cot_NL_CFG_SlowMidFast_Round3_ExpectedTime',],
    ['Cot_NL_CFG_SlowMidFast_Round4_Sorted_SlowCode', 'Cot_NL_CFG_SlowMidFast_Round4_Sorted_MidCode', 'Cot_NL_CFG_SlowMidFast_Round4_Sorted_FastCode', 'Cot_NL_CFG_SlowMidFast_Round4_Sorted_SlowTime', 'Cot_NL_CFG_SlowMidFast_Round4_Sorted_MidTime', 'Cot_NL_CFG_SlowMidFast_Round4_Sorted_FastTime', 'Cot_NL_CFG_SlowMidFast_Round4_ExpectedTime',],
    ]    
    args.slow_mid_fast_columns = slow_mid_fast_columns_full_list[args.iteration_round - 1]

    # =====================================================================================
    """ Select Model Based on Global Core Number """
    # CodeLlama     # Server
    if 100 <= core_index < 110:
        args.model_name = './CodeLlama-13b-Instruct-hf'
        args.start_point = (core_index-100) * 360
        args.end_point = args.start_point + 360
        if args.end_point > 599:
            args.end_point = 'End'
        if core_index == 109:
            args.start_point = 0
            args.end_point = 'End'
    
    # CodeLlama-34b-Instruct-hf
    elif 110 <= core_index < 120:
        args.model_name = './CodeLlama-34b-Instruct-hf'
        args.start_point = (core_index-110) * 180
        args.end_point = args.start_point + 180
        if args.end_point > 599:
            args.end_point = 'End'
        if core_index == 119:
            args.start_point = 0
            args.end_point = 'End'


    # ---------------------------------------------------------------

    # gemini-2.5-flash
    elif 230 <= core_index < 240:         # [230 ~ 234]
        args.model_name = 'gemini-2.5-flash'    # gemini-2.5-flash
        args.start_point = (core_index-230) * 150
        args.end_point = args.start_point + 150
        if args.end_point > 550:
            args.end_point = 'End'
        if core_index == 239:
            args.start_point = 0
            args.end_point = 'End'

    # ---------------------------------------------------------------

    # gpt-3.5-turbo
    elif 310 <= core_index < 320:
        args.model_name = 'gpt-3.5-turbo-0125'
        args.start_point = (core_index-310) * 150
        args.end_point = args.start_point + 150
        if args.end_point > 550:
            args.end_point = 'End'
        if core_index == 319:
            args.start_point = 0
            args.end_point = 'End'
  



    # =============================================================================
    # deepseek-reasoner
    elif 510 <= core_index < 520:
        args.model_name = 'deepseek-reasoner'
        args.start_point = (core_index-510) * 200
        args.end_point = args.start_point + 200
        if args.end_point > 950:
            args.end_point = 'End'
        if core_index == 519:
            args.start_point = 0
            args.end_point = 'End'
    

    
    # =================================================================================================
    """ Modify Save Path Based on Model """
    if ("CodeLlama" in args.model_name) and ("13b" in args.model_name):
        args.generated_df_path = f"{args.generated_df_path}_CodeLlama13B"
    elif ("CodeLlama" in args.model_name) and ("34b" in args.model_name):
        args.generated_df_path = f"{args.generated_df_path}_CodeLlama34B"

    elif ("gemini" in args.model_name) or ("Gemini" in args.model_name):
        args.generated_df_path = f"{args.generated_df_path}_Gemini"


    elif args.model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125"]:
        args.generated_df_path = f"{args.generated_df_path}_GPT3"
        args.return_log_probs = True

    elif args.model_name == "deepseek-reasoner":
        args.generated_df_path = f"{args.generated_df_path}_DeepSeekV32"

    
    
    # =================================================================================================
    """ # Dynamically construct module path, import module """
    module = importlib.import_module(f"Prompt_Template.{args.prompt_template_name}")
    Prompt_Dict = getattr(module, 'Prompt_Dict')  # Get Prompt Dictionary from module
    if ('_Py' in args.baseline_df_path) and ('_Cpp' not in args.baseline_df_path):
        args.input_role_play = Prompt_Dict['Role_Play_Python']
        args.input_operation_command = Prompt_Dict['Operation_Command_Python']
        args.input_io_format = Prompt_Dict['IO_Format']
    elif ('_Cpp' in args.baseline_df_path) and ('_Py' not in args.baseline_df_path):
        args.input_role_play = Prompt_Dict['Role_Play_Cpp']
        args.input_operation_command = Prompt_Dict['Operation_Command_Cpp']
        args.input_io_format = Prompt_Dict['IO_Format']

    # ---------------------------------------------------------------------------------------------------------
    _template_name = args.prompt_template_name.replace('__My_Adopted', '').replace('__', '_')

    root_directory =  os.path.dirname(args.generated_df_path)
    current_save_name = args.generated_df_path.split('\\')[-1].split('/')[-1]
    _current_name = f"/{current_save_name}__{_template_name}_{args.task_description}"
    _current_name = _current_name.replace(r'\PIE_', r'\args__PIE_').replace(r'/PIE_', r'/args__PIE_').replace(r'\DB_', r'\args__DB_').replace(r'/DB_', r'/args__DB_')
    _current_name = _current_name.replace(r'\Peace_', r'\args__Peace_').replace(r'/Peace_', r'/args__Peace_')
    args.log_path = root_directory + _current_name
    args.generated_df_path = f"{args.generated_df_path}__{_template_name}_{args.task_description}"


    # =================================================================================================
    # Number of generated codes = batch size * repeat times. One of them must be 1.
    assert args.num_generated_codes == args.batch_size * args.repeat_times
    assert (args.batch_size == 1) or (args.repeat_times == 1)



    # =================================================================================================
    """ # Set Temperature """
    if args.temperature == -404:
        if args.num_generated_codes == 1:
            args.temperature = 0.01           # Default for both models
        elif args.num_generated_codes >= 5:
            args.temperature = 0.7
        
    args.max_length = 1024   

    # ----------------------------------------
    args.core_index = core_index


    # ----------------------------------------
    """ Repository Level """
    if "_Repo_Level" in args.prompt_template_name:
        args.slow_function_head_column = "Slow_Code_Function_Head"
        args.io_test_column = "Test_Case_Function"

    return args


# #####################################################################################################################🔖💡✅🟨
def main(core_number=-1):
    args = hyperparameter_processing_func(core_number)
    print(f"✅✅✅=======================Starting Execution, Iteration Round: {args.iteration_round} | Core Index: {args.core_index} | Model Name: {args.model_name} | Generated DF Path: {args.generated_df_path}")
    args, low_efficiency_code_list, io_unit_test_dict_list, fast_cfg_list, generated_nl_desc_list, slow_mid_fast_three_codes_list, slow_code_function_head_list = import_data_func(args)
    use_multiprocess_code_generation_func(args, low_efficiency_code_list, io_unit_test_dict_list, fast_cfg_list, generated_nl_desc_list, slow_mid_fast_three_codes_list, slow_code_function_head_list)
    merge_generated_data_to_df_func(args)



# #####################################################################################################################🔖💡✅🟨
def hyperparameter_processing_func(core_number):
    args = get_hyperparameters()
    args = set_extra_parameters(args)
    args_dict = vars(args)
    print(pprint.pformat(args_dict))
    os.makedirs(f"{args.generated_df_path}", exist_ok=True)
    # Save args dictionary
    with open(f"{args.log_path}.txt", "w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=4)   # indent specifies indentation spaces, writing JSON to file in multiple lines

    return args

# #####################################################################################################################🔖💡✅🟨
def import_data_func(args):

    df = pd.read_csv(args.baseline_df_path)
    
    # -------------------------------------------
    low_efficiency_code_list = df[args.slow_code_column].tolist()
    io_unit_test_dict_list = []
    fast_cfg_list = []
    generated_nl_desc_list = []
    slow_mid_fast_three_codes_list = [[], [], []]
    slow_code_function_head_list = []
    if ('Use_IO' in args.prompt_template_name) or ('Public_IO' in args.prompt_template_name) or ('Merge_IO' in args.prompt_template_name) or ('Use_Test_Function' in args.prompt_template_name):
        io_unit_test_dict_list = df[args.io_test_column].tolist()
    if 'CFG' in args.prompt_template_name:
        fast_cfg_list = df[args.cfg_column_name].tolist()
    if 'Use_NL' in args.prompt_template_name:
        generated_nl_desc_list = df[args.nl_column].tolist()
    if 'Use_Slow_Mid_Fast_Time' in args.prompt_template_name:
        slow_mid_fast_three_codes_list = [df[col].tolist() for col in args.slow_mid_fast_columns]
    elif 'Use_Slow_Mid_Fast' in args.prompt_template_name:
        slow_mid_fast_three_codes_list = [df[col].tolist() for col in args.slow_mid_fast_columns[:3]]
    if 'Repo_Level' in args.prompt_template_name:
        slow_code_function_head_list = df[args.slow_function_head_column].tolist()


    
    if args.end_point == 'End':
        args.end_point = len(low_efficiency_code_list)

    return args, low_efficiency_code_list, io_unit_test_dict_list, fast_cfg_list, generated_nl_desc_list, slow_mid_fast_three_codes_list, slow_code_function_head_list


# #####################################################################################################################🔖💡✅🟨
def use_multiprocess_code_generation_func(args, low_efficiency_code_list, io_unit_test_dict_list, fast_cfg_list, generated_nl_desc_list, slow_mid_fast_three_codes_list, slow_code_function_head_list):
    # ==================================================================================================================================================================
    total_index_list = range(args.start_point, args.end_point)
    # Multiprocess parallel generation. Use multiple API keys (multi-threading/multi-processing) to process tasks in parallel.
    multiprocess_index_list = np.array_split(total_index_list, num_threads)   # Split the problem list into 'num_keys' parts.  

    # ------------------------------------------------------
    total_multiprocess_dict_list = []
    for repeat_count in range(args.repeat_times):
        multiprocess_dict_list = []
        for process_id in range(num_threads):
            temp_dict={}
            temp_dict['args'] = deepcopy(args)
            temp_dict['repeat_count'] = repeat_count
            temp_dict['key_index'] = process_id
            temp_dict['PIE_index_list'] = multiprocess_index_list[process_id]
            temp_dict['low_efficiency_code_list'] = low_efficiency_code_list
            temp_dict['io_unit_test_dict_list'] = io_unit_test_dict_list
            temp_dict['fast_cfg_list'] = fast_cfg_list
            temp_dict['generated_nl_desc_list'] = generated_nl_desc_list
            temp_dict['slow_mid_fast_three_codes_list'] = slow_mid_fast_three_codes_list
            temp_dict['slow_code_function_head_list'] = slow_code_function_head_list
            multiprocess_dict_list.append(temp_dict)
            
        total_multiprocess_dict_list.append(multiprocess_dict_list)

    # ------------------------------------------------------
    """
    Use multiprocessing.Pool (Process Pool) to process tasks in parallel.
    pool.map() will start a process for each data block in the chunked data
    """
    for repeat_count in range(args.repeat_times):
        # Parallel quantity equals key quantity. Automatically cleans up when leaving 'with' (equivalent to terminate()+join()) 
        with ThreadPoolExecutor(max_workers=num_threads) as thread_pool:   # Recommended: Use 'with' for automatic cleanup
        # with multiprocessing.Pool(processes=num_threads) as process_pool:
            list(thread_pool.map(process_imported_data_func, total_multiprocess_dict_list[repeat_count]))
        time.sleep(10)  # Wait 10 seconds to avoid API rate limits from submitting too fast
        




# #####################################################################################################################🔖💡✅🟨
def process_imported_data_func(multiprocess_dict_list):

    args = multiprocess_dict_list['args']
    repeat_count = multiprocess_dict_list['repeat_count']
    key_index = multiprocess_dict_list['key_index']
    PIE_index_list = multiprocess_dict_list['PIE_index_list']
    low_efficiency_code_list = multiprocess_dict_list['low_efficiency_code_list']
    io_unit_test_dict_list = multiprocess_dict_list['io_unit_test_dict_list']
    fast_cfg_list = multiprocess_dict_list['fast_cfg_list']
    generated_nl_desc_list = multiprocess_dict_list['generated_nl_desc_list']
    slow_mid_fast_three_codes_list = multiprocess_dict_list['slow_mid_fast_three_codes_list']
    slow_code_function_head_list = multiprocess_dict_list['slow_code_function_head_list']

    print(f"### Starting Process ID: {key_index}. Repeat Count: {repeat_count}. Processing {len(PIE_index_list)} items. [Start Index: {PIE_index_list[0]} ~ End Index: {PIE_index_list[-1]}]")

    for PIE_index in tqdm(PIE_index_list):

        index_str = str(PIE_index).zfill(4)

        if os.path.exists(f"{args.generated_df_path}/{index_str}.txt") or os.path.exists(f"{args.generated_df_path}/{index_str}_{repeat_count}.txt"):
            continue

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        """ First enter ablation determination """
        if "_Ablation_Remove_NL_" in args.prompt_template_name:
            input_problem_str = args.input_operation_command.format(  Slow_program = slow_mid_fast_three_codes_list[0][PIE_index].strip(),
                                                                    Medium_program = slow_mid_fast_three_codes_list[1][PIE_index].strip(),
                                                                    Fast_program = slow_mid_fast_three_codes_list[2][PIE_index].strip(),
                                                                    Slow_program_Time = slow_mid_fast_three_codes_list[3][PIE_index],
                                                                    Medium_program_Time = slow_mid_fast_three_codes_list[4][PIE_index],
                                                                    Fast_program_Time = slow_mid_fast_three_codes_list[5][PIE_index],
                                                                    Expected_Time = slow_mid_fast_three_codes_list[6][PIE_index],
                                                                    Refer_Fast_CFG = fast_cfg_list[PIE_index].strip() 
                                                                    )
        elif ("_Ablation_Remove_IO_" in args.prompt_template_name) and ('Generate_Code' in args.prompt_template_name):
            input_problem_str = args.input_operation_command.format(  Slow_program = slow_mid_fast_three_codes_list[0][PIE_index].strip(),
                                                                    Medium_program = slow_mid_fast_three_codes_list[1][PIE_index].strip(),
                                                                    Fast_program = slow_mid_fast_three_codes_list[2][PIE_index].strip(),
                                                                    Slow_program_Time = slow_mid_fast_three_codes_list[3][PIE_index],
                                                                    Medium_program_Time = slow_mid_fast_three_codes_list[4][PIE_index],
                                                                    Fast_program_Time = slow_mid_fast_three_codes_list[5][PIE_index],
                                                                    Expected_Time = slow_mid_fast_three_codes_list[6][PIE_index],
                                                                    Code_Function_Description = generated_nl_desc_list[PIE_index].strip(), 
                                                                    Refer_Fast_CFG = fast_cfg_list[PIE_index].strip() 
                                                                    )
        elif "_Ablation_Remove_CFG_" in args.prompt_template_name:
            input_problem_str = args.input_operation_command.format(  Slow_program = slow_mid_fast_three_codes_list[0][PIE_index].strip(),
                                                                    Medium_program = slow_mid_fast_three_codes_list[1][PIE_index].strip(),
                                                                    Fast_program = slow_mid_fast_three_codes_list[2][PIE_index].strip(),
                                                                    Slow_program_Time = slow_mid_fast_three_codes_list[3][PIE_index],
                                                                    Medium_program_Time = slow_mid_fast_three_codes_list[4][PIE_index],
                                                                    Fast_program_Time = slow_mid_fast_three_codes_list[5][PIE_index],
                                                                    Expected_Time = slow_mid_fast_three_codes_list[6][PIE_index],
                                                                    Code_Function_Description = generated_nl_desc_list[PIE_index].strip(), 
                                                                    )
        elif ("_Ablation_Remove_All_" in args.prompt_template_name):
            input_problem_str = args.input_operation_command.format(   Slow_program = low_efficiency_code_list[PIE_index].strip(), )


        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        # """ Enter Repository Level Determination """
        elif "_Repo_Level_COT_" in args.prompt_template_name:
            input_problem_str = args.input_operation_command.format(   Slow_program = low_efficiency_code_list[PIE_index].strip(), 
                                                                     Code_Function_Head = slow_code_function_head_list[PIE_index].strip(), 
                                                                     )
            
        elif "_Generate_NL_Repo_Level_" in args.prompt_template_name:
            input_problem_str = args.input_operation_command.format(   Slow_program = low_efficiency_code_list[PIE_index].strip(), 
                                                                     Test_case = io_unit_test_dict_list[PIE_index].strip(), 
                                                                     )
        elif "_Generate_Code_Repo_Level_" in args.prompt_template_name:
            input_problem_str = args.input_operation_command.format(  Slow_program = slow_mid_fast_three_codes_list[0][PIE_index].strip(),
                                                                    Medium_program = slow_mid_fast_three_codes_list[1][PIE_index].strip(),
                                                                    Fast_program = slow_mid_fast_three_codes_list[2][PIE_index].strip(),
                                                                    Slow_program_Time = slow_mid_fast_three_codes_list[3][PIE_index],
                                                                    Medium_program_Time = slow_mid_fast_three_codes_list[4][PIE_index],
                                                                    Fast_program_Time = slow_mid_fast_three_codes_list[5][PIE_index],
                                                                    Expected_Time = slow_mid_fast_three_codes_list[6][PIE_index],
                                                                    Code_Function_Description = generated_nl_desc_list[PIE_index].strip(), 
                                                                    Test_case = io_unit_test_dict_list[PIE_index].strip(), 
                                                                    Code_Function_Head = slow_code_function_head_list[PIE_index].strip(), 
                                                                    )
            input_problem_str = input_problem_str.replace("Medium function (Execution time pass ms): \n```python\npass\n```\n\n", "").replace("Fast function (Execution time pass ms): \n```python\npass\n```\n\n", "")
            input_problem_str = input_problem_str.replace("Medium function (Execution time pass ms): \n```cpp\npass\n```\n\n", "").replace("Fast function (Execution time pass ms): \n```cpp\npass\n```\n\n", "")




        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Generate Description
        elif 'Generate_NL' in args.prompt_template_name:
            input_problem_str = args.input_operation_command.format( Slow_program = low_efficiency_code_list[PIE_index].strip() )
        # Generate Code
        elif ('Generate_Code' in args.prompt_template_name) and ('Use_Slow_Mid_Fast_Time' in args.prompt_template_name) and ('CFG' in args.prompt_template_name):
            input_problem_str = args.input_operation_command.format(  Slow_program = slow_mid_fast_three_codes_list[0][PIE_index].strip(),
                                                                    Medium_program = slow_mid_fast_three_codes_list[1][PIE_index].strip(),
                                                                    Fast_program = slow_mid_fast_three_codes_list[2][PIE_index].strip(),
                                                                    Slow_program_Time = slow_mid_fast_three_codes_list[3][PIE_index],
                                                                    Medium_program_Time = slow_mid_fast_three_codes_list[4][PIE_index],
                                                                    Fast_program_Time = slow_mid_fast_three_codes_list[5][PIE_index],
                                                                    Expected_Time = slow_mid_fast_three_codes_list[6][PIE_index],
                                                                    Code_Function_Description = generated_nl_desc_list[PIE_index].strip(), 
                                                                    Refer_Fast_CFG = fast_cfg_list[PIE_index].strip() 
                                                                    )
            
        # Generate Code
        elif ('Generate_Code' in args.prompt_template_name) and ('Use_Slow_Mid_Fast' in args.prompt_template_name) and ('CFG' in args.prompt_template_name):
            input_problem_str = args.input_operation_command.format(  Slow_program = slow_mid_fast_three_codes_list[0][PIE_index].strip(),
                                                                    Medium_program = slow_mid_fast_three_codes_list[1][PIE_index].strip(),
                                                                    Fast_program = slow_mid_fast_three_codes_list[2][PIE_index].strip(),
                                                                    Code_Function_Description = generated_nl_desc_list[PIE_index].strip(), 
                                                                    Refer_Fast_CFG = fast_cfg_list[PIE_index].strip() 
                                                                    )   
            input_problem_str = input_problem_str.replace("Medium Code: \n```python\npass\n```\n\n", "").replace("Fast Code: \n```python\npass\n```\n\n", "")
        # Generate Code
        elif ('Generate_Code' in args.prompt_template_name) and ('Use_Slow_Mid_Fast' in args.prompt_template_name):
            input_problem_str = args.input_operation_command.format(  Slow_program = slow_mid_fast_three_codes_list[0][PIE_index].strip(),
                                                                    Medium_program = slow_mid_fast_three_codes_list[1][PIE_index].strip(),
                                                                    Fast_program = slow_mid_fast_three_codes_list[2][PIE_index].strip(),
                                                                    Code_Function_Description = generated_nl_desc_list[PIE_index].strip(), 
                                                                    )   
        # Generate Code
        elif ('Generate_Code' in args.prompt_template_name):
            input_problem_str = args.input_operation_command.format(   Slow_program = low_efficiency_code_list[PIE_index].strip(), 
                                                                     Code_Function_Description = generated_nl_desc_list[PIE_index].strip(), 
                                                                     Refer_Fast_CFG = fast_cfg_list[PIE_index].strip() )   

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Remove invalid code segments
        input_problem_str = input_problem_str.replace("Medium Code (Execution time pass ms): \n```python\npass\n```\n\n", "").replace("Fast Code (Execution time pass ms): \n```python\npass\n```\n\n", "")
        input_problem_str = input_problem_str.replace("Medium Code (Execution time pass ms): \n```cpp\npass\n```\n\n", "").replace("Fast Code (Execution time pass ms): \n```cpp\npass\n```\n\n", "")

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------
        if ('Use_IO' in args.prompt_template_name) or ('Add_Public_IO' in args.prompt_template_name) or ('Add_Merged_IO' in args.prompt_template_name):
            io_test_dict = eval(io_unit_test_dict_list[PIE_index])
            io_input_list = io_test_dict['inputs']
            io_output_list = io_test_dict['outputs']

            io_test_str = ""
            for i in range(len(io_input_list)):
                io_input = "'" + io_input_list[i].strip() + "'"
                io_output = "'" + io_output_list[i].strip() + "'"
                
                io_test_str = io_test_str + args.input_io_format.format( IO_ID=i+1, IO_Input=io_input, IO_Output=io_output )

            # --------------------------------------------------------------------------------------------------------------------
            # Generate Overall Description
            if 'Generate_NL' in args.prompt_template_name:
                input_problem_str = input_problem_str.replace("### Please follow the above instructions", f"{io_test_str}### Please follow the above instructions")
            # Generate Code
            elif ('Generate_Code' in args.prompt_template_name) and ('Use_Slow_Mid_Fast' in args.prompt_template_name):
                input_problem_str = input_problem_str.replace("### Test Case:", f"### Test Case:\n{io_test_str}")
            # Generate Code
            elif ('Generate_Code' in args.prompt_template_name):
                input_problem_str = input_problem_str.replace("### Please follow the above", f"{io_test_str}### Please follow the above")


        # -------------------------------------------------------------------------------------------------------------------------
        for _ in range(5):
            try:
                use_api_generate_data_func(args, input_problem_str, index_str, key_index, repeat_count)
                break
            except Exception as e:
                print(f"❌❌❌ Error when calling API to generate data: {e}")
                if 'gpt' in args.model_name:
                    time.sleep(random.randint(30, 90))  # Wait to avoid API rate limiting due to too frequent requests
                else:
                    time.sleep(random.randint(240, 900))  # Wait to avoid API rate limiting due to too frequent requests




# #####################################################################################################################🔖💡✅🟨
def use_api_generate_data_func(args, input_problem_str, index_str, key_index, repeat_count):
    assert args.batch_size == 1 or repeat_count == 0

    # ------------------------------------------------------------------------------------------------------------
    if args.model_name == "CodeGeneration2/CodeLlama-34b-Instruct-hf":
        model_response_text_list = CodeLlama_Deepinfra_Function(key_index=key_index, 
                                                model_name=args.model_name,   
                                                input_role_play=args.input_role_play, 
                                                input_problem_text=input_problem_str, 
                                                num_generations=args.batch_size, 
                                                temperature=args.temperature, 
                                                is_output_prompt=args.is_output_prompt,
                                                )
        save_to_text_file_func(args, index_str, repeat_count, model_response_text_list)


    # ------------------------------------------------------------------------------------------------------------
    elif "gemini" in args.model_name:
        model_response_text_list, avg_log_probs_list = Gemini_Official_Function(key_index=key_index, 
                                                            model_name=args.model_name,   
                                                            input_role_play=args.input_role_play, 
                                                            input_problem_text=input_problem_str, 
                                                            num_generations=args.batch_size, 
                                                            temperature=args.temperature, 
                                                            max_length="No Limit", 
                                                            thinking_budget=-404,           
                                                            return_log_probs=False,
                                                            is_output_prompt=args.is_output_prompt,
                                                            )
        save_to_text_file_func(args, index_str, repeat_count, model_response_text_list, avg_log_probs_list,)


    # ------------------------------------------------------------------------------------------------------------
    elif args.model_name in ["gpt-3.5-turbo-0125", ]:
        model_response_text_list, avg_log_probs_list = ChatGPT_Function(platform = "OpenAI_Official",                
                                                                key_index = key_index, 
                                                                model_name = args.model_name,    
                                                                input_role_play=args.input_role_play, 
                                                                input_problem_text=input_problem_str, 
                                                                num_generations = args.batch_size, 
                                                                temperature = args.temperature, 
                                                                max_length = args.max_length, 
                                                                return_log_probs=args.return_log_probs, 
                                                                is_output_prompt=args.is_output_prompt,
                                                                )
        if args.return_log_probs == False:
            avg_log_probs_list = []
        save_to_text_file_func(args, index_str, repeat_count, model_response_text_list, avg_log_probs_list,)

    
    # -------------------------------------------------------------------------------------------------------------
    elif args.model_name in ["deepseek-reasoner",]:
        model_response_text_list, model_return_dict, chain_of_thought_text, avg_log_probs = DeepSeek_Function( platform = "DeepSeek_Official",                
                                                                                                        key_index = key_index, 
                                                                                                        model_name = args.model_name,    
                                                                                                        input_role_play = args.input_role_play, 
                                                                                                        input_problem_text = input_problem_str, 
                                                                                                        temperature = args.temperature, 
                                                                                                        max_length = args.max_length, 
                                                                                                        is_output_prompt = args.is_output_prompt,
                                                                                                        )
        save_to_text_file_func(args, index_str, repeat_count, model_response_text_list, avg_log_probs_list=[avg_log_probs], chain_of_thought_text_list=[chain_of_thought_text])



# #####################################################################################################################🔖💡✅🟨
def save_to_text_file_func(args, index_str, repeat_count, model_response_text_list, avg_log_probs_list=[], chain_of_thought_text_list=[]):
    def generate_text_func(text_index, use_list_index):
        with open(f"{args.generated_df_path}/{index_str}_{text_index}.txt", 'w', encoding='UTF-8', newline='\n') as f:
            f.write(model_response_text_list[use_list_index])
        if avg_log_probs_list and (avg_log_probs_list[use_list_index] != ''):
            with open(f"{args.generated_df_path}/{index_str}_{text_index}_avg_log_probs.txt", 'w', encoding='UTF-8', newline='\n') as f:
                f.write(str(avg_log_probs_list[use_list_index]))
        if chain_of_thought_text_list and (chain_of_thought_text_list[use_list_index] != ""):
            with open(f"{args.generated_df_path}/{index_str}_{text_index}_chain_of_thought.txt", 'w', encoding='UTF-8', newline='\n') as f:
                f.write(chain_of_thought_text_list[use_list_index])
                
    # ---------------------------------------------------------------------------------
    if args.batch_size == 1:
        assert len(model_response_text_list) == args.batch_size == 1, "When batch_size is 1, length of model_response_text_list should be 1"
        generate_text_func(repeat_count, 0)
    elif repeat_count == 0:
        assert len(model_response_text_list) == args.batch_size, "When repeat_count is 0, length of model_response_text_list should equal batch_size"
        for i in range(args.batch_size):
            generate_text_func(i, i)



# #####################################################################################################################🔖💡✅🟨
def merge_generated_data_to_df_func(args):
    original_baseline_df_name = args.baseline_df_path.split('\\')[-1].split('/')[-1]
    df_new_name = args.generated_df_path.split('\\')[-1].split('/')[-1]
    save_table_path = args.baseline_df_path.replace(original_baseline_df_name, f"{df_new_name}.csv")

    if "Generate_NL" in args.prompt_template_name:
        merge_generated_io_descriptions_func(args.baseline_df_path, args.generated_df_path, save_table_path, args.save_column_prefix)
    elif "Generate_Code" in args.prompt_template_name:
        merge_to_df_fast_code_func(args.baseline_df_path, args.generated_df_path, save_table_path, args.save_column_prefix)
        







# #####################################################################################################################🔖💡✅🟨
if __name__ == "__main__":
    print("############################################################################################################🔖💡✅🟨 Start File Large_Model_API_Generation__Latest5.py")
    main()