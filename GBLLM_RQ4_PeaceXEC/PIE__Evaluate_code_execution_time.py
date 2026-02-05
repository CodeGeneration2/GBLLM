# -*- coding: utf-8 -*-
# print(f"### :\n{}")
# #####################################################################################################################🔖💡✅🟨

import os
from tqdm import tqdm
import json
import pandas as pd
from API__PIE_sandbox import create_cmd_process_and_run_python_code_io_unit_test_func

import statistics
import pprint
import argparse



# #####################################################################################################################🔖💡✅🟨❌
DEBUG = False


TEST_COLUMN_NAME_LIST = [


]


# #####################################################################################################################🔖💡✅🟨


# #####################################################################################################################🔖💡✅🟨
def get_hyperparameters():

    parser = argparse.ArgumentParser()

    parser.add_argument("--test_io_type",     default='(Public) (Private)', type=str)

    parser.add_argument("--iteration_round",       default=-404, type=int)
    parser.add_argument("--ignore_count",       default=1, type=int)
    parser.add_argument("--run_count_excluding_ignore", default=26, type=int)
    parser.add_argument("--dataset_path",     default=r"code_data_table", type=str)
    parser.add_argument("--save_set_path",     default=r"io_completed_code_data_table", type=str)
    parser.add_argument("--is_last_test",   default=False, type=bool)
    parser.add_argument("--accuracy_threshold",     default=0.999, type=float)

    args = parser.parse_args()

    return args


# #####################################################################################################################🔖💡✅🟨
def set_extra_parameters(args):
    # -----------------------------------------------------------------------------------------------
    global TEST_IO_TYPE, ITERATION_ROUND, IGNORE_COUNT, RUN_COUNT_EXCLUDING_IGNORE, DATASET_PATH, SAVE_SET_PATH, IS_LAST_TEST, ACCURACY_THRESHOLD
    TEST_IO_TYPE = args.test_io_type
    ITERATION_ROUND = args.iteration_round
    IGNORE_COUNT = args.ignore_count
    RUN_COUNT_EXCLUDING_IGNORE = args.run_count_excluding_ignore
    DATASET_PATH = args.dataset_path
    SAVE_SET_PATH = args.save_set_path
    IS_LAST_TEST = args.is_last_test
    ACCURACY_THRESHOLD = args.accuracy_threshold

    default_test_column_name_list = [
    ['Cot_NL_CFG_SlowMidFastTime_Round2_G5__Predict_Fast_code_1', 'Cot_NL_CFG_SlowMidFastTime_Round2_G5__Predict_Fast_code_2', 'Cot_NL_CFG_SlowMidFastTime_Round2_G5__Predict_Fast_code_3', 'Cot_NL_CFG_SlowMidFastTime_Round2_G5__Predict_Fast_code_4', 'Cot_NL_CFG_SlowMidFastTime_Round2_G5__Predict_Fast_code_5',],
    ['Cot_NL_CFG_SlowMidFastTime_Round2_G5__Predict_Fast_code_1', 'Cot_NL_CFG_SlowMidFastTime_Round2_G5__Predict_Fast_code_2', 'Cot_NL_CFG_SlowMidFastTime_Round2_G5__Predict_Fast_code_3', 'Cot_NL_CFG_SlowMidFastTime_Round2_G5__Predict_Fast_code_4', 'Cot_NL_CFG_SlowMidFastTime_Round2_G5__Predict_Fast_code_5',],
    ['Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code_1', 'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code_2', 'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code_3', 'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code_4', 'Cot_NL_CFG_SlowMidFastTime_Round3_G5__Predict_Fast_code_5',],
    ]

    if ITERATION_ROUND >= 1:
        global TEST_COLUMN_NAME_LIST
        TEST_COLUMN_NAME_LIST = default_test_column_name_list[ITERATION_ROUND - 1]


# #####################################################################################################################🔖💡✅🟨
def main():
    args = get_hyperparameters()
    set_extra_parameters(args)
    print(f"###=======================Start Execution, Iteration Round: {args.iteration_round}")
    print(pprint.pformat(vars(args)))
    if os.path.exists(SAVE_SET_PATH) == False:
        multi_dataset_traversal_func(args)


# #####################################################################################################################🔖💡✅🟨
def multi_dataset_traversal_func(args):

    # -----------------------------
    if SAVE_SET_PATH.endswith('.csv') == False:   # If save path is not a CSV file
        find_current_save_number_func(SAVE_SET_PATH)
        os.makedirs(f'{SAVE_SET_PATH}', exist_ok=True)

    if DATASET_PATH.endswith('.csv') == False:   # If dataset path is not a CSV file
        df_name_list = os.listdir(DATASET_PATH)
        assert len(df_name_list) == 1, f"Dataset path {DATASET_PATH} must contain exactly one file"

        df_name = df_name_list[0]  # Get the first file name
        df = pd.read_csv(f'{DATASET_PATH}/{df_name}')        # Read CSV file  
    else:
        df_name = DATASET_PATH.split('/')[-1].split('\\')[-1]
        df = pd.read_csv(f'{DATASET_PATH}')        # Read CSV file

    # -------------------------------------------------
    """ Set Hyperparameters """
    global DATASET, PROGRAMMING_LANGUAGE, ENVIRONMENT_NAME
    if ('PIE' in df_name) and ('DB' not in df_name) and ('ECCO' not in df_name):
        DATASET = 'PIE'
    elif ('DB' in df_name) and ('PIE' not in df_name) and ('ECCO' not in df_name):
        DATASET = 'DB'
    elif ('ECCO' in df_name) and ('PIE' not in df_name) and ('DB' not in df_name):
        DATASET = 'ECCO'
    if ('_Py' in df_name) and ('_Cpp' not in df_name):
        PROGRAMMING_LANGUAGE = 'python'
    elif ('_Cpp' in df_name) and ('_Py' not in df_name):
        PROGRAMMING_LANGUAGE = 'cpp'
    if (IS_LAST_TEST) and (DATASET == 'PIE'):
        ENVIRONMENT_NAME = 'py39'
    elif (IS_LAST_TEST) and (DATASET == 'DB'):
        ENVIRONMENT_NAME = 'py3137'    
    elif (IS_LAST_TEST) and (DATASET == 'ECCO'):
        ENVIRONMENT_NAME = 'py310'    
    else:
        ENVIRONMENT_NAME = 'py3137'


    # ==========================================================================================================================================
    if IS_LAST_TEST:
        if '(Public)' in TEST_IO_TYPE:
            used_io_type = 'Public_IO_unit_tests__Dedup'     # !!!!!!!!!!!!!!!! Look at results, consider changing to Public_IO_unit_tests__Dedup
            print(f"### Using params, Dataset:{DATASET}, Check env is:{ENVIRONMENT_NAME}, Lang:({PROGRAMMING_LANGUAGE}), ### DF Name:{df_name}, ## Used IO Type:{used_io_type}")
            df = single_df_traversal_func(df_name, df, used_io_type)
        if '(Generate)' in TEST_IO_TYPE:
            used_io_type = TEST_IO_TYPE.split(':')[-1].strip()
            print(f"### Using params, Dataset:{DATASET}, Check env is:{ENVIRONMENT_NAME}, Lang:({PROGRAMMING_LANGUAGE}), ### DF Name:{df_name}, ## Used IO Type:{used_io_type}")
            df = single_df_traversal_func(df_name, df, used_io_type)
        if '(Private)' in TEST_IO_TYPE:
            used_io_type = 'Hide_IO_unit_tests'
            print(f"### Using params, Dataset:{DATASET}, Check env is:{ENVIRONMENT_NAME}, Lang:({PROGRAMMING_LANGUAGE}), ### DF Name:{df_name}, ## Used IO Type:{used_io_type}")
            df = single_df_traversal_func(df_name, df, used_io_type)
    # Not the last test
    else:
        if '(Public)' in TEST_IO_TYPE:
            used_io_type = 'Public_IO_unit_tests__Dedup'
            print(f"### Using params, Dataset:{DATASET}, Check env is:{ENVIRONMENT_NAME}, Lang:({PROGRAMMING_LANGUAGE}), ### DF Name:{df_name}, ## Used IO Type:{used_io_type}")
            df = single_df_traversal_func(df_name, df, used_io_type)
        if '(Generate)' in TEST_IO_TYPE:
            used_io_type = TEST_IO_TYPE.split(':')[-1].strip()
            print(f"### Using params, Dataset:{DATASET}, Check env is:{ENVIRONMENT_NAME}, Lang:({PROGRAMMING_LANGUAGE}), ### DF Name:{df_name}, ## Used IO Type:{used_io_type}")
            df = single_df_traversal_func(df_name, df, used_io_type)
        if '(Private)' in TEST_IO_TYPE:
            used_io_type = 'Hide_IO_unit_tests__Dedup'
            print(f"### Using params, Dataset:{DATASET}, Check env is:{ENVIRONMENT_NAME}, Lang:({PROGRAMMING_LANGUAGE}), ### DF Name:{df_name}, ## Used IO Type:{used_io_type}")
            df = single_df_traversal_func(df_name, df, used_io_type)

    # -------------------------------------------------------------------
    if ITERATION_ROUND == -404:
        os.remove(f'{DATASET_PATH}/{df_name}')  # Delete original dataset file


# #####################################################################################################################🔖💡✅🟨
def single_df_traversal_func(df_name, df, used_io_type):
 
    for col_name in TEST_COLUMN_NAME_LIST:
        if ('_Ori' not in col_name) and ('_log_probs' not in col_name):
            df = traverse_single_column_func(df, prediction_column_name=col_name, save_name=col_name, used_io_type=used_io_type)

    # Save modified data to new CSV file
    if 'PIE_Py_' in df_name:
        pie_number = df_name.split('PIE_Py_')[-1].split('_')[0]
    elif 'PIE_Cpp_' in df_name:
        pie_number = df_name.split('PIE_Cpp_')[-1].split('_')[0]
    elif 'DB_Py_' in df_name:
        pie_number = df_name.split('DB_Py_')[-1].split('_')[0]
    elif 'ECCO_Py_' in df_name:
        pie_number = df_name.split('ECCO_Py_')[-1].split('_')[0]
    

    # Create new DF file name
    new_pie_number = str(int(pie_number) + 1).zfill(3)
    original_name = df_name.split('__')[0].replace(pie_number, new_pie_number)

    if 'Public_IO_unit_tests' in used_io_type:
        new_name = f"{original_name}__Intermediate_Public_IO_Completed.csv"
    elif 'Hide_IO_unit_tests' in used_io_type:
        new_name = f"{original_name}__Hidden_IO_Completed.csv"
    elif ('Generate_IO' in used_io_type) or ('Gen_IO' in used_io_type):
        new_name = f"{original_name}__Intermediate_Gen_IO_Completed.csv"

    # Save modified data to new CSV file
    if SAVE_SET_PATH.endswith('.csv'):
        df.to_csv(f'{SAVE_SET_PATH}', index=False)
    else:
        df.to_csv(f'{SAVE_SET_PATH}/{new_name}', index=False)

    return df


# #################################################################################################################################################
def traverse_single_column_func(df, prediction_column_name, save_name, used_io_type):
    overall_io_pass_rate, io_pass_rate_list, execution_time_list, evaluation_result_list = traversal_func(df, prediction_column_name=prediction_column_name, used_io_type=used_io_type)
    print(f"### Column:{prediction_column_name}  ## IO Type:{used_io_type}  ## Overall IO Pass Rate:{overall_io_pass_rate}%")

    _result_dict = {}
    if IS_LAST_TEST and (PROGRAMMING_LANGUAGE == 'python'):
        if ('__Predict_Fast_code' in save_name) and (DATASET == 'PIE'):
            save_name = save_name.replace('__Predict_Fast_code', '_Py39__Predict_Fast_code')
        elif ('__Predict_Fast_code' in save_name) and (DATASET == 'ECCO'):
            save_name = save_name.replace('__Predict_Fast_code', '_Py310__Predict_Fast_code')
        elif (DATASET == 'PIE'):
            save_name = f'{save_name}_Py39'
        elif (DATASET == 'ECCO'):
            save_name = f'{save_name}_Py310'

    # ---------------------------------------------------------------------
    if 'Public_IO_unit_tests' in used_io_type:
        _result_dict[f'{save_name}__Public_IO_pass_rate_(%)'] = io_pass_rate_list
        _result_dict[f'{save_name}__Public_time(ms)'] = execution_time_list
        _result_dict[f'{save_name}__Public_Eval_Result'] = evaluation_result_list
    elif 'Hide_IO_unit_tests' in used_io_type:
        _result_dict[f'{save_name}__IO_pass_rate_(%)'] = io_pass_rate_list
        _result_dict[f'{save_name}__time(ms)'] = execution_time_list
        _result_dict[f'{save_name}__Eval_Result'] = evaluation_result_list
    elif used_io_type == 'Gen_IO_Code_Only_4_Instances__Merge_Public_IO':
        _result_dict[f'{save_name}__Gen_IO_pass_rate_(%)'] = io_pass_rate_list
        _result_dict[f'{save_name}__Gen_time(ms)'] = execution_time_list
        _result_dict[f'{save_name}__Gen_Eval_Result'] = evaluation_result_list
    elif used_io_type == 'Gen_IO_Code_Only_4_Instances':
        _result_dict[f'{save_name}__Initial_Gen_IO_pass_rate_(%)'] = io_pass_rate_list
        _result_dict[f'{save_name}__Initial_Gen_time(ms)'] = execution_time_list
        _result_dict[f'{save_name}__Initial_Gen_Eval_Result'] = evaluation_result_list
    elif used_io_type == 'Gen_IO_Code_Only_4_Instances_Initial__Repair_IO':
        _result_dict[f'{save_name}__Final_Gen_IO_pass_rate_(%)'] = io_pass_rate_list
        _result_dict[f'{save_name}__Final_Gen_time(ms)'] = execution_time_list
        _result_dict[f'{save_name}__Final_Gen_Eval_Result'] = evaluation_result_list
    
    elif used_io_type == 'Gen_IO_Code_Only_4_Instances__Merge_Public_IO_Del0':
        _result_dict[f'{save_name}__Gen_Del0_IO_pass_rate_(%)'] = io_pass_rate_list
        _result_dict[f'{save_name}__Gen_Del0_time(ms)'] = execution_time_list
        _result_dict[f'{save_name}__Gen_Del0_Eval_Result'] = evaluation_result_list
    elif 'Gen_IO_Code_Only_4_Instances__Repair_IO__Del0__Merge_Public_IO__5Limit' == used_io_type:
        _result_dict[f'{save_name}__Gen_5Limit_IO_pass_rate_(%)'] = io_pass_rate_list
        _result_dict[f'{save_name}__Gen_5Limit_time(ms)'] = execution_time_list
        _result_dict[f'{save_name}__Gen_5Limit_Eval_Result'] = evaluation_result_list
    elif 'Gen_IO' in used_io_type:
        _result_dict[f'{save_name}__Gen_IO_pass_rate_(%)'] = io_pass_rate_list
        _result_dict[f'{save_name}__Gen_time(ms)'] = execution_time_list
        _result_dict[f'{save_name}__Gen_Eval_Result'] = evaluation_result_list
    elif 'Generate_IO' in used_io_type:
        _result_dict[f'{used_io_type}__IO_pass_rate_(%)'] = io_pass_rate_list
        _result_dict[f'{used_io_type}__time(ms)'] = execution_time_list
        _result_dict[f'{used_io_type}__Eval_Result'] = evaluation_result_list

    # Merge once after loop finishes
    df = pd.concat([df, pd.DataFrame(_result_dict, index=df.index)], axis=1)
    # Save modified data to new CSV file
    if (ITERATION_ROUND == -404) and ('Hide_IO_unit_tests' in used_io_type):
        df.to_csv(f'{SAVE_SET_PATH}/IO_Temp_{prediction_column_name}.csv', index=False)

    return df


# #################################################################################################################################################
def traversal_func(df, prediction_column_name, used_io_type):

    prediction_code_list = df[prediction_column_name].tolist()
    io_test_list = df[used_io_type].tolist()

    io_pass_rate_list = []
    execution_time_list = []
    evaluation_result_list = []

    
    for i in tqdm(range(len(prediction_code_list))):  # Iterate through each row

        # Check code compliance
        tested_code_str = str(prediction_code_list[i])
        # code_compliance = check_code_func(tested_code_str)
        # if code_compliance != True:
        #     print(f"Code not compliant: {code_compliance}")
        

        # Evaluate code
        io_dict = eval(io_test_list[i])
        if io_dict['inputs'] == []:
            io_pass_rate_list.append(0)
            execution_time_list.append(1234567890)
            evaluation_result_list.append({'IO_Pass_Results': [], 'Time_Usage_List': [], 'Time_Standard_Deviation': 1234567890, 'Time_List_Unit': 'milliseconds_ms = e-3 seconds', 'Error_Type': ['No IO'], 'Subprocess_Return_Code': 0, 'Subprocess_Stderr': ''})
            continue    
        
        
        # ----------------------------------------------------------------------------------------------------------------
        if 'Public_IO_unit_tests' in used_io_type or 'Hide_IO_unit_tests' in used_io_type or 'Gen_IO_Code_Only_4_Instances__Merge_Public_IO' in used_io_type or 'Gen_IO_Code_Only_4_Instances__Merge_Public_IO_Del0' in used_io_type:
            # io_pass_rate, execution_time, evaluation_result_dict = io_evaluation_func(tested_code_str, io_dict, 0, 1, i)
            # if RUN_COUNT_EXCLUDING_IGNORE > 1 and io_pass_rate > ACCURACY_THRESHOLD:
            io_dict2 = eval(io_test_list[i])
            io_pass_rate, execution_time, evaluation_result_dict = io_evaluation_func(tested_code_str, io_dict2, IGNORE_COUNT, RUN_COUNT_EXCLUDING_IGNORE, pie_index=i, stat_value='Mean')
        
        elif used_io_type in ['Gen_IO_Code_Only_4_Instances', 'Gen_IO_Code_Only_4_Instances_Initial__Repair_IO']:
            io_dict2 = eval(io_test_list[i])
            io_pass_rate, execution_time, evaluation_result_dict = io_evaluation_func(tested_code_str, io_dict2, 0, 1, pie_index=i, stat_value='Mean')


        io_pass_rate_list.append(io_pass_rate)
        execution_time_list.append(execution_time)
        evaluation_result_list.append(evaluation_result_dict)
        

    # ----------------------------------------------------------------------------------------------------------------
    # if 'Public_IO_unit_tests' in used_io_type or 'Hide_IO_unit_tests' in used_io_type:
    overall_io_pass_rate = round(len([v for v in io_pass_rate_list if v > ACCURACY_THRESHOLD]) / len(io_pass_rate_list) * 100, 2)
    # elif 'Generate_IO' in used_io_type or 'Gen_IO' in used_io_type:
    #     print(f"### IO_pass_rate_list: {set(io_pass_rate_list)}")
    #     overall_io_pass_rate = round(statistics.mean(io_pass_rate_list) * 100, 2)

    return overall_io_pass_rate, io_pass_rate_list, execution_time_list, evaluation_result_list


# #####################################################################################################################🔖💡✅🟨"stdout" in code line or "stderr"
def check_code_func(tested_code_str):
    risk_keywords = ['setrecursionlimit(', 'stack_size(', "if __name__ == '__main__':", ' is not ', ' is ', 'stdout', 'stderr',
             ", file=sys.stdout", ", file=stdout", ", output=sys.stdout", ", output=stdout", ", file=sys.stderr", ", file=stderr", ", output=sys.stderr", ", output=stderr",
             "sys.stdin.readline", "stdin.readline", "sys.stdin.buffer.readline", "stdin.buffer.readline",
             "sys.stdout.write", "stdout.write", "sys.__stdout__.write", "__stdout__.write", "sys.stderr.write", "stderr.write", "sys.__stderr__.write", "__stderr__.write",
             'return sys.stdout.flush()', 
             "IOWrapper(", "FastIO(", "StringIO(", "BytesIO(", "FastStdout(", ".close(", "stdout", "stderr", 
             'open(',
             "threading", "thread", "multiprocessing", "asyncio", "queue.Queue(", "ProcessPoolExecutor", "concurrent", "fork(", "subprocess.run("]
    
    for risk_keyword in risk_keywords:
        if risk_keyword in tested_code_str and 'open(0' not in tested_code_str:
            return risk_keyword
    return True


# #####################################################################################################################🔖💡✅🟨
def io_evaluation_func(tested_code_str, io_dict, ignore_count, run_count_excluding_ignore, pie_index=0, stat_value='Mean'):
    """
    Evaluation_Result_Dict = {"IO_Pass_Results": io_pass_results_list,
                        "Time_Usage_List": time_usage_list,
                        "Error_Label": 1,
                        "Error_Info_Dict": {},
                        "Tested_Code": original_tested_code_str,
                        "Time_List_Unit": 'microseconds_μs = e-6 seconds',
                        "Code_Output_List": code_output_list}
    """
    # Use json.dump() to save dictionary as JSON file
    with open("temp_Code.txt", "w", encoding='utf-8') as f:
        f.write(tested_code_str.strip())
    with open("temp_test_IO.json", 'w', encoding='utf-8') as f:
        json.dump(io_dict, f, ensure_ascii=False, indent=4)
    # try:
    evaluation_result_dict = create_cmd_process_and_run_python_code_io_unit_test_func(  tested_code_path="temp_Code.txt", 
                                                                                        io_file="temp_test_IO.json", 
                                                                                        ignore_count=ignore_count, 
                                                                                        run_count_excluding_ignore=run_count_excluding_ignore,
                                                                                        programming_language=PROGRAMMING_LANGUAGE,
                                                                                        pie_index=pie_index,
                                                                                        )
    os.remove("temp_Code.txt")
    os.remove("temp_test_IO.json")

    if DEBUG:
        print(f"### Evaluation Result Dict: {evaluation_result_dict}")
        pprint.pprint(evaluation_result_dict)
        return 0, 1234567890, evaluation_result_dict


    io_pass_single_list = evaluation_result_dict['IO_Pass_Results']
    io_time_double_list = evaluation_result_dict['Time_Usage_List']
    assert len(io_pass_single_list) == len(io_time_double_list), f"IO Pass Single List and IO Time Double List length mismatch {len(io_pass_single_list)} != {len(io_time_double_list)}"

    io_pass_rate = statistics.mean(io_pass_single_list)
    assert 0 <= io_pass_rate <= 1, f"IO Pass Rate Range Error {io_pass_rate}"

    # Handle IO Pass Rate == 0
    if io_pass_rate < 0.0001:
        return io_pass_rate, 1234567890, evaluation_result_dict

    # Handle IO Pass Rate == 1
    elif io_pass_rate > 0.9999:
        if stat_value == 'Median':
            current_time_list = [statistics.median(current_io_time_list) for current_io_time_list in io_time_double_list]
        elif stat_value == 'Mean':
            current_time_list = [statistics.mean(current_io_time_list) for current_io_time_list in io_time_double_list]
        execution_time = round(statistics.mean(current_time_list), 2)
        return io_pass_rate, execution_time, evaluation_result_dict

    # --------------------------------------------------------------------------------------------
    # """ Mainly execution time only statistics when IO is 1 """
    else:
        # 1. Find positions in list1 equal to 1, and filter both lists:
        new_io_pass_results_list = [x for x in io_pass_single_list if x == 1]
        if new_io_pass_results_list == []:
            return 0, 1234567890, evaluation_result_dict

        new_execution_time_double_list = [io_time_double_list[i] for i, x in enumerate(io_pass_single_list) if x == 1]
        if stat_value == 'Median':
            current_time_list = [statistics.median(current_io_time_list) for current_io_time_list in new_execution_time_double_list]
        elif stat_value == 'Mean':
            current_time_list = [statistics.mean(current_io_time_list) for current_io_time_list in new_execution_time_double_list]
        execution_time = round(statistics.mean(current_time_list), 2)

        return io_pass_rate, execution_time, evaluation_result_dict

        
# #####################################################################################################################🔖💡✅🟨
def find_current_save_number_func(_save_path):
    global SAVE_SET_PATH
    history_save_name_list = os.listdir('./')
    history_number_name_list = []
    for name in history_save_name_list:
        if name.startswith(SAVE_SET_PATH):
            history_number_name_list.append(int(name.split(SAVE_SET_PATH)[-1]))
    current_number = str(max(history_number_name_list, default=0) + 1).zfill(2)
    SAVE_SET_PATH = f"{_save_path}{current_number}"




# #################################################################################################################################################
if __name__ == '__main__':
    print("############################################################################################################🔖💡✅🟨 Start File T8__PIE__Evaluate_Code_Execution_Time__Latest6.py")
    main()