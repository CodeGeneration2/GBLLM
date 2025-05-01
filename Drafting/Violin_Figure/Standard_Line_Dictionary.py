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


def main():
    evaluate_top_1_function()


def evaluate_top_1_function():
    # Read CSV file  
    df = pd.read_csv(f'{dataset_path}')

    original_slow_code_IO_list = df['input__IO_pass_rate_(%)'].tolist()
    original_slow_code_time_list = df['input__time(ms)'].tolist()

    human_code_IO_list = df['target__IO_pass_rate_(%)'].tolist()
    human_code_time_list = df['target__time(ms)'].tolist()
    total_standard_dict = get_standard_line_list_for_violin_plot_function(original_slow_code_IO_list, original_slow_code_time_list, human_code_IO_list, human_code_time_list)

    with open(f"standard_line_dict.txt", 'w', encoding='utf-8') as f:
        f.write(str(total_standard_dict))



def get_standard_line_list_for_violin_plot_function(slow_IO_list, slow_time_list, human_IO_list, human_time_list):
    assert len(slow_IO_list) == len(slow_time_list) == len(human_IO_list) == len(human_time_list)

    slow_standard_line_dict = {}
    for i in range(len(slow_IO_list)):
        if slow_IO_list[i] > 0.999:
           slow_standard_line_dict[i] = slow_time_list[i]
        else:
            slow_standard_line_dict[i] = 1234567890

    human_standard_line_dict = {}
    for i in range(len(human_IO_list)):
        if human_IO_list[i] > 0.999:
           human_standard_line_dict[i] = human_time_list[i]
        else:
            human_standard_line_dict[i] = 1234567890

    ten_percent_human_standard_line_dict = {}
    for i in range(len(human_IO_list)):
        if human_IO_list[i] > 0.999:
           ten_percent_human_standard_line_dict[i] = round(human_time_list[i] / 1.1 , 4)
        else:
            ten_percent_human_standard_line_dict[i] = 1234567890
    
    # ---------------------------------
    total_standard_dict = {}
    total_standard_dict['slow_standard_line'] = slow_standard_line_dict
    total_standard_dict['human_standard_line'] = human_standard_line_dict
    total_standard_dict['ten_percent_human_standard_line'] = ten_percent_human_standard_line_dict

    return total_standard_dict


if __name__ == '__main__':
    main()
