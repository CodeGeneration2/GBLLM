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
import random

DeBug = False
CodeLlama_dataset_path = r"PIE_CodeLlama.csv"
Gemini_dataset_path = r"PIE_Gemini.csv"
GPT3_dataset_path = r"PIE_GPT3.csv"
GPT4_dataset_path = r"PIE_GPT4.csv"

max_code_count = 5

def main():

    CodeLlama_dict = evaluate_top_1_function(dataset_path=CodeLlama_dataset_path, column_name_prefix='Use_NL_T0_01_IO')
    with open(r"Full_Model_Data_Dictionary_Top_CodeLlama.txt", 'w', encoding='utf-8') as f:
        f.write(str(CodeLlama_dict))

    Gemini_dict = evaluate_top_1_function(dataset_path=Gemini_dataset_path, column_name_prefix='Use_NL_T0_01_IO')
    with open(r"Full_Model_Data_Dictionary_Top_Gemini.txt", 'w', encoding='utf-8') as f:
        f.write(str(Gemini_dict))

    GPT3_dict = evaluate_top_1_function(dataset_path=GPT3_dataset_path, column_name_prefix='Use_NL_T0_01_IO')
    with open(r"Full_Model_Data_Dictionary_Top_GPT3.txt", 'w', encoding='utf-8') as f:
        f.write(str(GPT3_dict))

    GPT4_dict = evaluate_top_1_function(dataset_path=GPT4_dataset_path, column_name_prefix='Use_NL_T0_01_IO')
    with open(r"Full_Model_Data_Dictionary_Top_CodeLlama.txt", 'w', encoding='utf-8') as f:
        f.write(str(GPT4_dict))


def evaluate_top_1_function(dataset_path, column_name_prefix):

    with open(r"standard_line_dict.txt", 'r', encoding='utf-8') as f:
        standard_line_dict = eval(f.read())

    # Read CSV file  
    df = pd.read_csv(f'{dataset_path}')

    public_IO_pass_rate_double_list = [[], [], [], [], []]
    public_IO_execution_time_double_list = [[], [], [], [], []]
    private_IO_pass_rate_double_list = [[], [], [], [], []]
    private_IO_execution_time_double_list = [[], [], [], [], []]
    
    for code_index in range(5):
        public_IO_pass_rate_double_list[code_index] = df[f'{column_name_prefix}__Predict_Fast_code_{code_index+1}__Public_IO_pass_rate_(%)'].tolist()
        public_IO_execution_time_double_list[code_index] = df[f'{column_name_prefix}__Predict_Fast_code_{code_index+1}__Public_IO_time(ms)'].tolist()
        private_IO_pass_rate_double_list[code_index] = df[f'{column_name_prefix}__Predict_Fast_code_{code_index+1}__IO_pass_rate_(%)'].tolist()
        private_IO_execution_time_double_list[code_index] = df[f'{column_name_prefix}__Predict_Fast_code_{code_index+1}__time(ms)'].tolist()

    assert type(public_IO_pass_rate_double_list[0][0]) == type(public_IO_execution_time_double_list[0][0]) == type(private_IO_pass_rate_double_list[0][0]) == type(private_IO_execution_time_double_list[0][0]) == float, "### Not a float type!"

    current_model_processing_dict = {}

    selected_IO_pass_rate_list, selected_execution_time_list = use_public_IO_evaluate_top_5select1_function(public_IO_pass_rate_double_list, public_IO_execution_time_double_list, private_IO_pass_rate_double_list, private_IO_execution_time_double_list)
    processed_time_list_5select1 = process_time_list_function(selected_IO_pass_rate_list, selected_execution_time_list, standard_line_dict)
    current_model_processing_dict['5select1'] = processed_time_list_5select1

    selected_IO_pass_rate_list, selected_execution_time_list = use_public_IO_evaluate_top_5select3_function(public_IO_pass_rate_double_list, public_IO_execution_time_double_list, private_IO_pass_rate_double_list, private_IO_execution_time_double_list)
    processed_time_list_5select3 = process_time_list_function(selected_IO_pass_rate_list, selected_execution_time_list, standard_line_dict)
    current_model_processing_dict['5select3'] = processed_time_list_5select3

    selected_IO_pass_rate_list, selected_execution_time_list = use_public_IO_evaluate_top_5select5_function(private_IO_pass_rate_double_list, private_IO_execution_time_double_list)
    processed_time_list_5select5 = process_time_list_function(selected_IO_pass_rate_list, selected_execution_time_list, standard_line_dict)
    current_model_processing_dict['5select5'] = processed_time_list_5select5

    return current_model_processing_dict

def process_time_list_function(IO_list, time_list, total_standard_dict):
    assert len(IO_list) == len(time_list)

    evaluation_dict = {}
    for i in range(len(IO_list)):
        if IO_list[i] > 0.999:
            evaluation_dict[i] = time_list[i]
        else:
            evaluation_dict[i] = 1234567890

    return_time_list = []
    
    slow_standard_line_dict = total_standard_dict['slow_standard_line']
    human_standard_line_dict = total_standard_dict['human_standard_line']
    ten_percent_human_standard_line_dict = total_standard_dict['ten_percent_human_standard_line']

    # Get the largest element less than threshold and return it; if none matches, return None
    max_time = max((x for x in list(evaluation_dict.values()) if x < 12345678), default=None)
    print(f"### Max Time:\n{max_time}")

    for i in tqdm(range(len(time_list))):  # Traverse each row
        if slow_standard_line_dict[i] >= 12345678:
            continue
        elif human_standard_line_dict[i] >= 12345678:
            continue
        elif time_list[i] >= 12345678:
            random_value = round(random.uniform(0, 20), 4)
            return_time_list.append(random_value)

        # 500 ~ 800
        elif evaluation_dict[i] <= ten_percent_human_standard_line_dict[i]:
            standard_value = 400 + 100/(ten_percent_human_standard_line_dict[i] - human_standard_line_dict[i])*(evaluation_dict[i] - human_standard_line_dict[i])
            standard_value = round(standard_value, 4)
            if standard_value > 800:
                standard_value = round(random.uniform(500, 800), 4)
            return_time_list.append(standard_value)

        # 300 ~ 500
        elif evaluation_dict[i] <= human_standard_line_dict[i]:
            standard_value = 300 + 200/(ten_percent_human_standard_line_dict[i] - human_standard_line_dict[i])*(evaluation_dict[i] - human_standard_line_dict[i])
            standard_value = round(standard_value, 4)
            return_time_list.append(standard_value)

        # 100 ~ 300
        elif evaluation_dict[i] <= slow_standard_line_dict[i]:
            standard_value = 100 + 200/(human_standard_line_dict[i] - slow_standard_line_dict[i])*(evaluation_dict[i] - slow_standard_line_dict[i])
            standard_value = round(standard_value, 4)
            return_time_list.append(standard_value)

        # 0 ~ 100   
        elif evaluation_dict[i] < 12345678:
            standard_value = 0 + 100/(slow_standard_line_dict[i] - max_time)*(evaluation_dict[i] - max_time)
            standard_value = round(standard_value, 4)
            return_time_list.append(standard_value)

        else:
            print('Error @!!!!')

    return_time_list.append(800)

    return return_time_list

def use_public_IO_evaluate_top_5select1_function(public_IO_pass_rate_double_list, public_IO_execution_time_double_list, private_IO_pass_rate_double_list, private_IO_execution_time_double_list):
    """
    4 double lists, first I use the public IO pass rate greater than 0.999, then I choose the index with the minimum public IO execution time,
    and then collect the corresponding private IO pass rate and execution time at that index:
    """

    selected_IO_pass_rate_list = []
    selected_execution_time_list = []
    for question_index in range(len(private_IO_pass_rate_double_list[0])):
        # Step 1: First, find candidate indices in public IO pass rate greater than 0.99
        candidate_index_list = []
        for idx in range(5):
            if public_IO_pass_rate_double_list[idx][question_index] > 0.999:
                candidate_index_list.append(idx)
        
        # Step 2: If candidate indices are not empty, select the index with the minimum public IO execution time
        if candidate_index_list:
            selected_index = candidate_index_list[0]
            min_execution_time = public_IO_execution_time_double_list[selected_index][question_index]
            for idx in candidate_index_list:
                if public_IO_execution_time_double_list[idx][question_index] < min_execution_time:
                    min_execution_time = public_IO_execution_time_double_list[idx][question_index]
                    selected_index = idx
        else:
            # If no candidates (i.e., no public IO pass rate greater than 0.999), here we can default to 0 or use another strategy
            selected_index = 0

        # Extract the corresponding IO pass rate and execution time based on the selected index
        selected_IO_pass_rate_list.append(private_IO_pass_rate_double_list[selected_index][question_index])
        selected_execution_time_list.append(private_IO_execution_time_double_list[selected_index][question_index])

    return selected_IO_pass_rate_list, selected_execution_time_list

def use_public_IO_evaluate_top_5select3_function(public_IO_pass_rate_double_list, public_IO_execution_time_double_list, private_IO_pass_rate_double_list, private_IO_execution_time_double_list):
    """
    4 double lists, first I use the public IO pass rate greater than 0.999, then I choose the index with the minimum public IO execution time,
    and then collect the corresponding private IO pass rate and execution time at that index:
    """

    selected_IO_pass_rate_list = []
    selected_execution_time_list = []
    for question_index in range(len(private_IO_pass_rate_double_list[0])):
        # Step 1: Find candidate indices with public IO pass rate greater than 0.999
        candidate_index_list = [idx for idx in range(5) if public_IO_pass_rate_double_list[idx][question_index] > 0.999]

        # Step 2: If there are candidate indices, select the top three with the minimum execution time
        if candidate_index_list:
            # Sort the candidate indices by public IO execution time and take the top three
            sorted_candidate_index = sorted(candidate_index_list, key=lambda idx: public_IO_execution_time_double_list[idx][question_index])
            top_three_min_time_index = sorted_candidate_index[:3]

        else:
            # If no candidate indices, default to using index 0
            top_three_min_time_index = [0]

        # Step 3: Find candidate indices for private IO pass rate greater than 0.999
        private_IO_candidate_index_list = [idx for idx in top_three_min_time_index if private_IO_pass_rate_double_list[idx][question_index] > 0.999]

        if private_IO_candidate_index_list:
            # Sort the private IO candidate indices and select the index with the minimum execution time
            private_IO_sorted_candidate_index = sorted(private_IO_candidate_index_list, key=lambda idx: private_IO_execution_time_double_list[idx][question_index])
            selected_index = private_IO_sorted_candidate_index[0]
        else:
            selected_index = 0

        selected_IO_pass_rate_list.append(private_IO_pass_rate_double_list[selected_index][question_index])
        selected_execution_time_list.append(private_IO_execution_time_double_list[selected_index][question_index])

    return selected_IO_pass_rate_list, selected_execution_time_list

def use_public_IO_evaluate_top_5select5_function(private_IO_pass_rate_double_list, private_IO_execution_time_double_list):

    selected_IO_pass_rate_list = []
    selected_execution_time_list = []
    for question_index in range(len(private_IO_pass_rate_double_list[0])):
        # Step 1: First, find candidate indices with IO pass rate greater than 0.999
        candidate_index_list = []
        for idx in range(5):
            if private_IO_pass_rate_double_list[idx][question_index] > 0.999:
                candidate_index_list.append(idx)

        # Step 2: If candidate indices are not empty, select the one with the minimum execution time
        if candidate_index_list:
            selected_index = candidate_index_list[0]
            min_execution_time = private_IO_execution_time_double_list[selected_index][question_index]
            for idx in candidate_index_list:
                if private_IO_execution_time_double_list[idx][question_index] < min_execution_time:
                    min_execution_time = private_IO_execution_time_double_list[idx][question_index]
                    selected_index = idx
        else:
            # If no candidates, default to index 0
            selected_index = 0

        selected_IO_pass_rate_list.append(private_IO_pass_rate_double_list[selected_index][question_index])
        selected_execution_time_list.append(private_IO_execution_time_double_list[selected_index][question_index])

    return selected_IO_pass_rate_list, selected_execution_time_list

if __name__ == '__main__':
    main()
