import json
import pprint
import statistics
import openai
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use only one GPU.


# Note: Several libraries are imported more than once. They are kept as in the
# original code, but they can be simplified in practical development.
import json
from tqdm import tqdm
import copy
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
# import tiktoken
import time

# Import the custom evaluation module for calculating code execution efficiency.
# from code_efficiency_calculator import calculate_code_execution_efficiency

from tqdm import tqdm

from API__single_generation__5 import deepseek_dual_platform_function  # Display progress bars in the terminal.
from API__single_generation__5 import codellama_deepinfra_function
from API__single_generation__5 import gemini_official_function
from API__single_generation__5 import gemini_cloudmist_api_function
from API__single_generation__5 import chatgpt_three_platforms_function
from API__single_generation__5 import deepseek_dual_platform_function
from API__single_generation__5 import codellama_server_standard_model_load_function
from API__single_generation__5 import codellama_server_standard_inference_function


from API__PIE_sandbox__9 import run_python_cpp_io_unit_tests_in_new_shell_process
import pandas as pd
import copy  # Ensure that the copy module is imported.

import uuid



# #####################################################################################################################
DEBUG = False
dataset_path = r"./PIE_Cpp_009_CodeLlama34B__X.csv"
output_dataset_path = r""
parallel_count = 1








# #####################################################################################################################
# Define the list of models to be tested.
if "CodeLlama13B" in dataset_path:
    model_name = "./CodeLlama-13b-Instruct-hf"
    codellama_tokenizer, codellama_model = codellama_server_standard_model_load_function(model_name=model_name)
elif "CodeLlama34B" in dataset_path:
    model_name = "./CodeLlama-34b-Instruct-hf"
    codellama_tokenizer, codellama_model = codellama_server_standard_model_load_function(model_name=model_name)
elif "DeepSeekV32" in dataset_path:
    model_name = "deepseek-ai/DeepSeek-V3.2-Exp"
elif "Gemini" in dataset_path:
    model_name = "gemini-2.5-flash-nothinking"
elif "GPT3" in dataset_path:
    model_name = "gpt-3.5-turbo-1106"



# #####################################################################################################################
def main():

    global programming_language
    if "_Py_" in dataset_path:
        programming_language = 'python'
    elif "_Cpp_" in dataset_path:
        programming_language = 'cpp'

    # Dictionary used to collect global statistical metrics.
    global_metrics_dict = {
        "overhead": [],
        "memory_usage": [],
        "execution_time": [],
        "max_memory_peak": [],
        "correct": []
    }

    # Read the dataset to be optimized.
    """ # In Python memory, this is a list. Each element in the list is a dictionary
    # representing one task entry.
    [
        {
            "markdown_description": "Write a Python function `add(a, b)` that returns the sum of two numbers.",
            "small_test_cases": "assert add(1, 2) == 3\nassert add(-1, 1) == 0",
            "completion": "def add(a, b):\n    return a + b"
        },
        {
            "markdown_description": "Write a function to find the maximum element in a list.",
            "small_test_cases": "assert find_max([1, 5, 3]) == 5",
            "completion": "def find_max(lst):\n    max_val = lst[0]\n    for num in lst:\n        if num > max_val:\n            max_val = num\n    return max_val"
        }
    ]
    """
    # with open(f"./EffiBench_{model_name}.json", "r") as f:
    #     full_dataset = json.load(f)
    # ==================== Replacement region starts ====================
    df = pd.read_csv(dataset_path)
    # df = df[:2]

    if DEBUG:
        df = df[:4]

    # Convert the DataFrame into the List[Dict] base format required by the code.
    full_dataset = []
    for _, row in df.iterrows():
        # 1. Convert all data in the current row into a dictionary. This preserves
        # all columns and values from the original table.
        row_dict = row.to_dict()

        # 2. Add or overwrite the three core keys required for code execution.
        row_dict["completion"] = str(row['input'])
        row_dict["small_test_cases"] = str(row['Public_IO_unit_tests__Dedup'])
        row_dict["markdown_description"] = "None"

        # Optional: remove old duplicate keys if you do not want to keep them.
        row_dict.pop('input', None)
        row_dict.pop('Public_IO_unit_tests__Dedup', None)

        full_dataset.append(row_dict)


    # Perform five rounds of iterative optimization.
    for i in range(5):
        print(f"\n\nCurrent iteration: {i}. Model: {model_name}. Dataset: {dataset_path}")
        total_memory_usage = 0
        total_execution_time = 0
        total_max_memory_peak = 0
        successful_execution_count = 0  # Record the number of successfully executed code entries.

        # ====== Step 1: Evaluate the execution efficiency of the current code in parallel. ======
        print(f"    Round {i}: Step 1: Evaluate the execution efficiency of the current code in parallel.")
        with ThreadPoolExecutor(max_workers=parallel_count) as executor:
            # Submit all evaluation tasks.
            futures = {executor.submit(update_dataset_entry, entry): entry for entry in full_dataset}

            # Collect evaluation results.
            for future in tqdm(as_completed(futures), total=len(full_dataset), desc="Running code evaluation"):
                entry = futures[future]
                result = future.result()

                # If the code successfully updates performance data, whether for the original code or the new code.
                if result["update"]:
                    overhead, memory_usage, execution_time, max_memory_peak, executable = result["data"]
                    # Write the latest performance metrics into the data dictionary.
                    entry["overhead"] = overhead
                    entry["memory_usage"] = memory_usage
                    entry["execution_time"] = execution_time
                    entry["max_memory_peak"] = max_memory_peak
                    entry["executable"] = executable

                    successful_execution_count += 1
                    total_memory_usage += memory_usage
                    total_execution_time += execution_time
                    total_max_memory_peak += max_memory_peak

                    # Finalize the optimization result: replace the official code with the temporary code.
                    if "tmp_completion" in entry.keys():
                        entry["completion"] = entry["tmp_completion"]
                else:
                    # If there is no update but the original code is executable, include the existing
                    # performance data in the total statistics.
                    if "executable" in entry.keys() and entry["executable"]:
                        successful_execution_count += 1
                        total_memory_usage += entry["memory_usage"]
                        total_execution_time += entry["execution_time"]
                        total_max_memory_peak += entry["max_memory_peak"]
                        # Ensure that tmp_completion is consistent with the original completion.
                        if "tmp_completion" in entry.keys():
                            entry["tmp_completion"] = entry["completion"]

        # ====== Step 2: Compute and record the global average metrics for this round. ======
        print(f"    Round {i}: Step 2: Compute and record the global average metrics for this round.")
        if successful_execution_count > 0:
            total_overhead = f"""
The total memory usage during the code execution is: {round(total_memory_usage/successful_execution_count, 2)} MB*s.
The total execution time is: {round(total_execution_time/successful_execution_count, 2)} s.
The maximum memory peak requirement is: {round(total_max_memory_peak/successful_execution_count, 2)} MB.
"""
            # Append the averaged performance metrics to the dictionary.
            global_metrics_dict["overhead"].append(total_overhead)
            global_metrics_dict["memory_usage"].append(round(total_memory_usage/successful_execution_count, 2))
            global_metrics_dict["execution_time"].append(round(total_execution_time/successful_execution_count, 2))
            global_metrics_dict["max_memory_peak"].append(round(total_max_memory_peak/successful_execution_count, 2))
            global_metrics_dict["correct"].append(successful_execution_count)
        else:
            print("No correct entries to calculate overall metrics.")

        print("    Successful execution count:", successful_execution_count)

        # ====== Step 3: Call the LLM in parallel to obtain the optimized code for the next round. ======
        print(f"    Round {i}: Step 3: Call the LLM ({model_name}) in parallel to obtain the optimized code for the next round.")
        with ThreadPoolExecutor(max_workers=parallel_count) as executor:
            # Pass copy.deepcopy(entry) to avoid directly modifying the original object in multiple threads.
            future_to_entry = {
                executor.submit(fetch_completion, copy.deepcopy(entry), model_name, idx % parallel_count): entry
                for idx, entry in enumerate(full_dataset)
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_entry), total=len(full_dataset)):
                entry = future_to_entry[future]
                try:
                    updated_entry = future.result()

                    # ==================== New logic: save the raw code generated each time. ====================
                    # After fetch_completion finishes, updated_entry["tmp_completion"] stores the exact raw code
                    # just produced by the model. Because i ranges from 0 to 4, i + 1 gives column names 1 to 5.
                    raw_code = updated_entry.get("tmp_completion", "")
                    updated_entry[f'EffiLearner_G1__Predict_Fast_code_{i+1}'] = raw_code
                    # ==================== End of new logic. ====================

                    # Find the index of the corresponding data entry and update the original dataset with the new data returned by the LLM.
                    idx = full_dataset.index(entry)
                    full_dataset[idx] = updated_entry
                except Exception as e:
                    print(repr(e))

    # ====== End of loop: perform the final result evaluation and statistics. ======
    total_memory_usage = 0
    total_execution_time = 0
    total_max_memory_peak = 0
    successful_execution_count = 0

    with ThreadPoolExecutor(max_workers=parallel_count) as executor:
        futures = {executor.submit(update_dataset_entry, entry): entry for entry in full_dataset}

        for future in as_completed(futures):
            entry = futures[future]
            result = future.result()
            if result["update"]:
                overhead, memory_usage, execution_time, max_memory_peak, executable = result["data"]
                entry["overhead"] = overhead
                entry["memory_usage"] = memory_usage
                entry["execution_time"] = execution_time
                entry["max_memory_peak"] = max_memory_peak
                entry["executable"] = executable
                if "tmp_completion" in entry.keys():
                    entry["completion"] = entry["tmp_completion"]
                successful_execution_count += 1
                total_memory_usage += memory_usage
                total_execution_time += execution_time
                total_max_memory_peak += max_memory_peak
            else:
                if "executable" in entry.keys() and entry["executable"]:
                    successful_execution_count += 1
                    total_memory_usage += entry["memory_usage"]
                    total_execution_time += entry["execution_time"]
                    total_max_memory_peak += entry["max_memory_peak"]
                    if "tmp_completion" in entry.keys():
                        entry["tmp_completion"] = entry["completion"]

    # Record the global statistics for the final round.
    if successful_execution_count > 0:
        total_overhead = f"""
The total memory usage during the code execution is: {round(total_memory_usage/successful_execution_count, 2)} MB*s.
The total execution time is: {round(total_execution_time/successful_execution_count, 2)} s.
The maximum memory peak requirement is: {round(total_max_memory_peak/successful_execution_count, 2)} MB.
"""
        global_metrics_dict["overhead"].append(total_overhead)
        global_metrics_dict["memory_usage"].append(round(total_memory_usage/successful_execution_count, 2))
        global_metrics_dict["execution_time"].append(round(total_execution_time/successful_execution_count, 2))
        global_metrics_dict["max_memory_peak"].append(round(total_max_memory_peak/successful_execution_count, 2))
        global_metrics_dict["correct"].append(successful_execution_count)

    # ====== Save the final optimized dataset and global metrics to a local JSON file. ======
    # with open(f"./EffiBench_{model_name}.json", "w") as f:
    #     json.dump(full_dataset, f, indent=4)

    # ====== Save the final optimized dataset back to a DataFrame and export it. ======
    # 1. Convert List[Dict] back to a DataFrame.
    result_df = pd.DataFrame(full_dataset)

    # 2. Optional: rename columns back to their original names.
    result_df.rename(columns={
        "completion": "input", 
        "small_test_cases": "Public_IO_unit_tests__Dedup"
    }, inplace=True)

    # 3. Save the DataFrame as a CSV file, or another table format if needed.
    # Automatically generate a new name.
    if output_dataset_path == '':
        prefix_path = '__'.join(dataset_path.split('__')[:-1])
        pie_number = dataset_path.split('__')[-2].split('_')[-2]
        prefix_path = prefix_path.replace(pie_number, str(int(pie_number)+1).zfill(3))
        generated_output_dataset_path = f"{prefix_path}__EffiLearner_generated.csv"
    else:
        generated_output_dataset_path = output_dataset_path
    # Save the modified data to a new CSV file.
    result_df.to_csv(f'{generated_output_dataset_path}', index=False)


    with open(f"./overhead_{model_name}.json", "w") as f:
        json.dump(global_metrics_dict, f, indent=4)

    print(f"Model {model_name} is done")








# #####################################################################################################################
def prompt_construction(task_description, test_case, completion, overhead_prompt):
    """
    Construct the prompt used to request code optimization from the LLM.

    Args:
        task_description (str): The specific task description.
        test_case (str): Test cases used to verify code correctness.
        completion (str): The original code. In this prompt template, it is inserted directly
            and also semantically used as the code to be optimized.
        overhead_prompt (str): The performance overhead analysis result of the current code.

    Returns:
        str: The fully concatenated prompt string.
    """
    prompt = f"""
Optimize the efficiency of the following Python code based on the task, test case, and overhead analysis provided. Ensure the optimized code can pass the given test case.

Task Description:
{task_description}

Test Case:
{test_case}

Original Code:
```python
{completion}
```

Overhead Analysis:
{overhead_prompt}

Optimization Rules:
- Encapsulate the optimized code within a Python code block (i.e., ```python\n[Your Code Here]\n```).
- Do not include the test case within the code block.
- Focus solely on code optimization; test cases are already provided.
- Ensure the provided test case passes with your optimized solution.
"""
    return prompt


# #####################################################################################################################
def calculate_execution_efficiency_metrics(entry):
    """
    Calculate the code execution efficiency metrics for a single data entry.

    Args:
        entry (dict): A data dictionary containing code and related information.

    Returns:
        tuple: (overhead information, memory usage, execution time, maximum memory peak, executable flag).
    """
    overhead, memory_usage, execution_time, max_memory_peak, executable = calculate_code_execution_efficiency(entry)

    return overhead, memory_usage, execution_time, max_memory_peak, executable



# #####################################################################################################################
# Function to fetch completion.
def fetch_completion(data_entry, model_name, parallel_index):
    """
    Call the API to obtain optimized code and temporarily store the result in the data dictionary.

    Args:
        data_entry (dict): A single task data entry.
        model_name (str): The name of the LLM model being used, such as "gpt-4".

    Returns:
        dict: The updated data entry, including the temporary result "tmp_completion".
    """
    # If no small test cases are provided, return the original data directly.
    if "small_test_cases" not in data_entry.keys():
        return data_entry

    # Obtain the current performance overhead information. If unavailable, mark execution as failed.
    if "overhead" not in data_entry.keys():
        overhead = "The code execution failed."
    else:
        overhead = data_entry["overhead"]

    test_case = data_entry["small_test_cases"]
    completion = data_entry["completion"]
    task_description = data_entry["markdown_description"]

    # Construct the request prompt.
    prompt = prompt_construction(task_description, test_case, completion, overhead)

    if programming_language == 'cpp':
        prompt = prompt.replace("Python", "C++").replace("python", "cpp")

    try:
        model_response_text_list = generate_data_with_api(
            "You are a code developer expert. Please generate only the code.",
            prompt,
            api_key_index=parallel_index,
            print_prompt=DEBUG,
        )


        # Store the LLM output in the temporary field "tmp_completion" for later validation.
        processed_code = postprocess_cot_generated_code_py_cpp(model_response_text_list[0], prefix=programming_language)
        data_entry["tmp_completion"] = processed_code

    except Exception as e:
        print("API error: call failed:", repr(e))
        # If the API call fails, record the error message.
        data_entry["tmp_completion"] = "API error: call failed"

    return data_entry






# #####################################################################################################################
def generate_data_with_api(role_prompt, input_question_str, api_key_index, print_prompt):

    # ------------------------------------------------------------------------------------------------------------
    if model_name in ["./CodeLlama-34b-Instruct-hf", "./CodeLlama-13b-Instruct-hf"]:
        model_response_text_list = codellama_server_standard_inference_function(
            model=codellama_model,
            role_prompt=role_prompt,
            input_question_text=input_question_str,
            generation_count=1,
            temperature=0.7,
            max_length=1024,
            print_prompt=print_prompt,
            token_table=codellama_tokenizer,
        )


    # ------------------------------------------------------------------------------------------------------------
    elif model_name == "CodeGeneration2/CodeLlama-34b-Instruct-hf":
        model_response_text_list = codellama_deepinfra_function(
            api_key_index=api_key_index,
            model_name=model_name,
            role_prompt=role_prompt,
            input_question_text=input_question_str,
            generation_count=1,
            temperature=0.7,
            print_prompt=print_prompt,
        )




    # ------------------------------------------------------------------------------------------------------------
    elif "gemini" in model_name:
        model_response_text_list = gemini_cloudmist_api_function(
            platform="CloudMist API",  # [CloudMist AI, OpenAI official, Close_AI]
            api_key_index=api_key_index,
            model_name=model_name,  # gpt-3.5-turbo-0125, gpt-4o, gpt-4-1106-preview, gemini-2.5-flash-thinking
            role_prompt=role_prompt,
            input_question_text=input_question_str,
            generation_count=1,  # Must be 1.
            temperature=0.7,  # Effective.
            max_length=1024,  # Ineffective.
            print_prompt=print_prompt,
        )


    # ------------------------------------------------------------------------------------------------------------
    elif False and "gemini" in model_name:
        model_response_text_list, average_log_probability_list = gemini_official_function(
            api_key_index=api_key_index,
            model_name=model_name,
            role_prompt=role_prompt,
            input_question_text=input_question_str,
            generation_count=1,
            temperature=0.7,
            max_length="no_limit",
            thinking_budget=-404,  # -1: enable dynamic thinking; 0: disable thinking; greater than 0: fixed thinking budget 8.
            return_log_probability=False,
            print_prompt=print_prompt,
        )




    # ------------------------------------------------------------------------------------------------------------
    elif model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "gpt-4o-mini", "gpt-4.1-nano"]:
        model_response_text_list, average_log_probability_list = chatgpt_three_platforms_function(
            platform="CloudMist API",  # [CloudMist AI, OpenAI official, Close_AI]
            api_key_index=api_key_index,
            model_name=model_name,  # gpt-3.5-turbo-0125, gpt-4o, gpt-4-1106-preview
            role_prompt=role_prompt,
            input_question_text=input_question_str,
            generation_count=1,
            temperature=0.7,
            max_length=1024,
            return_log_probability=False,
            print_prompt=print_prompt,
        )


    # -------------------------------------------------------------------------------------------------------------
    elif model_name in ["deepseek-chat", "deepseek-reasoner", "deepseek-ai/DeepSeek-V3.2-Exp"]:
        if model_name in ["deepseek-chat", "deepseek-reasoner"]:
            platform_name = "DeepSeek official"
        elif model_name in ["deepseek-ai/DeepSeek-V3.2-Exp"]:
            platform_name = "SiliconFlow API"
        model_response_text_list, full_model_response_dict, chain_of_thought_text, average_log_probability = deepseek_dual_platform_function(
            platform=platform_name,  # [DeepSeek official, SiliconFlow API]
            api_key_index=api_key_index,
            model_name=model_name,  # [deepseek-chat, deepseek-reasoner, deepseek-ai/DeepSeek-V3.2-Exp]
            role_prompt=role_prompt,
            input_question_text=input_question_str,
            temperature=0.7,
            max_length=1024,
            print_prompt=print_prompt,
        )

    return model_response_text_list



# #####################################################################################################################
def update_dataset_entry(single_data_dict):
    """
    Evaluate the code and decide whether to replace the old code in "completion"
    with the newly generated code in "tmp_completion".
    The criterion is that the code must run successfully and use less memory than before.

    Args:
        single_data_dict (dict): A data dictionary containing code and its execution status.

    Returns:
        dict: A dictionary containing the "update" Boolean state and the new performance data.
    """
    # If the current entry has no newly generated optimized code, evaluate the original code and return.
    if "tmp_completion" not in single_data_dict.keys():
        comprehensive_analysis_report, memory_usage, execution_time, peak_memory_usage, is_correct = calculate_code_execution_efficiency(single_data_dict)
        if is_correct:
            return {
                "update": True,
                "data": (comprehensive_analysis_report, memory_usage, execution_time, peak_memory_usage, is_correct)
            }
    else:
        # If newly generated optimized code exists, first back up the original code.
        original_completion = single_data_dict["completion"]
        # Replace the test subject with the newly generated code.
        single_data_dict["completion"] = single_data_dict["tmp_completion"]

        # Calculate the execution efficiency of the newly generated code.
        comprehensive_analysis_report, memory_usage, execution_time, peak_memory_usage, is_correct = calculate_code_execution_efficiency(single_data_dict)

        # Decide whether to keep the new code:
        # 1. If there was no memory data before and the new code is executable, keep it.
        if "memory_usage" not in single_data_dict.keys():
            return {
                "update": True,
                "data": (comprehensive_analysis_report, memory_usage, execution_time, peak_memory_usage, is_correct)
            }

        # 2. If the new code is executable and its execution time is lower than the original value, keep it.
        elif is_correct and execution_time < single_data_dict.get("memory_usage", float('inf')):
            return {
                "update": True,
                "data": (comprehensive_analysis_report, memory_usage, execution_time, peak_memory_usage, is_correct)
            }
        # 3. Otherwise, if the code has errors or is less efficient, roll back to the original code.
        else:
            single_data_dict["completion"] = original_completion

    return {"update": False}




# #####################################################################################################################
def calculate_code_execution_efficiency(single_data_dict):

    tested_code_str = single_data_dict["completion"]
    io_dict = eval(single_data_dict["small_test_cases"])

    io_pass_rate, execution_time, execution_memory, evaluation_result_dict = calculate_pie_code_execution_efficiency(
        tested_code_str,
        io_dict,
        ignored_runs=1,
        measured_runs=3,
        pie_index=0,
        statistic_value='mean'
    )

    return io_pass_rate, execution_memory, execution_time, execution_memory, io_pass_rate > 0.9999



# #####################################################################################################################
def calculate_pie_code_execution_efficiency(tested_code_str, io_dict, ignored_runs=1, measured_runs=3, pie_index=0, statistic_value='mean'):
    """
    evaluation_result_dict = {
        "io_pass_results": io_pass_result_list,
        "time_usage_lists": time_usage_lists,
        "memory_usage_lists": memory_usage_lists,
        "error_id": 1,
        "error_info_dict": {},
        "tested_code": original_tested_code_str,
        "time_list_unit": "microseconds_us_equals_e_minus_6_seconds",
        "code_output_list": code_output_list,
    }
    """
    # Generate a unique identifier.
    unique_id = uuid.uuid4().hex

    # Use json.dump() to save the dictionary as a JSON file.
    if programming_language == 'python':
        tested_code_path = f"temp_Code_{unique_id}.py"
    elif programming_language == 'cpp':
        tested_code_path = f"temp_Code_{unique_id}.cpp"
    with open(tested_code_path, "w", encoding='utf-8') as f:
        f.write(tested_code_str.strip())
    with open(f"temp_test_IO_{unique_id}.json", 'w', encoding='utf-8') as f:
        json.dump(io_dict, f, ensure_ascii=False, indent=4)


    evaluation_result_dict = run_python_cpp_io_unit_tests_in_new_shell_process(
        tested_code_path=tested_code_path,
        io_file=f"temp_test_IO_{unique_id}.json",
        ignored_runs=ignored_runs,
        measured_runs=measured_runs,
        programming_language=programming_language,
        pie_index=pie_index,
        timeout_seconds=9,
    )
    os.remove(tested_code_path)
    os.remove(f"temp_test_IO_{unique_id}.json")

    if DEBUG:
        print(f"### evaluation_result_dict: {evaluation_result_dict}")
        pprint.pprint(evaluation_result_dict)


    io_pass_single_list = evaluation_result_dict['io_pass_results']
    io_time_nested_list = evaluation_result_dict['time_usage_lists']
    io_memory_nested_list = evaluation_result_dict['memory_usage_lists']
    # Add memory extraction with get-based fault tolerance.
    # io_memory_nested_list = evaluation_result_dict.get(
    #     'memory_usage_lists',
    #     [[0.0] * len(sublist) for sublist in io_time_nested_list]
    # )

    assert len(io_pass_single_list) == len(io_time_nested_list), (
        f"The lengths of io_pass_single_list and io_time_nested_list are inconsistent: "
        f"{len(io_pass_single_list)} != {len(io_time_nested_list)}.\n"
        f"evaluation_result_dict: {evaluation_result_dict}"
    )

    io_pass_rate = statistics.mean(io_pass_single_list)
    assert 0 <= io_pass_rate <= 1, f"io_pass_rate is out of range: {io_pass_rate}"

    # Handle io_pass_rate == 0.
    if io_pass_rate < 0.0001:
        return io_pass_rate, 1234567890, 1234567890, evaluation_result_dict

    # Handle io_pass_rate == 1.
    elif io_pass_rate > 0.9999:
        if statistic_value == 'median':
            current_time_list = [statistics.median(current_io_time_list) for current_io_time_list in io_time_nested_list]
            current_memory_list = [statistics.median(current_io_memory_list) for current_io_memory_list in io_memory_nested_list]
        elif statistic_value == 'mean':
            current_time_list = [statistics.mean(current_io_time_list) for current_io_time_list in io_time_nested_list]
            current_memory_list = [statistics.mean(current_io_memory_list) for current_io_memory_list in io_memory_nested_list]
        execution_time = round(statistics.mean(current_time_list), 2)
        execution_memory = round(statistics.mean(current_memory_list), 2)
        return io_pass_rate, execution_time, execution_memory, evaluation_result_dict

    # --------------------------------------------------------------------------------------------
    # Mainly, execution time and memory are counted only for IO cases that pass.
    else:
        # 1. Find positions in the list that equal 1 and filter the lists.
        new_io_pass_result_list = [x for x in io_pass_single_list if x == 1]
        if new_io_pass_result_list == []:
            return 0, 1234567890, 1234567890, evaluation_result_dict

        new_runtime_nested_list = [io_time_nested_list[i] for i, x in enumerate(io_pass_single_list) if x == 1]
        new_memory_nested_list = [io_memory_nested_list[i] for i, x in enumerate(io_pass_single_list) if x == 1]

        if statistic_value == 'median':
            current_time_list = [statistics.median(current_io_time_list) for current_io_time_list in new_runtime_nested_list]
            current_memory_list = [statistics.median(current_io_memory_list) for current_io_memory_list in new_memory_nested_list]
        elif statistic_value == 'mean':
            current_time_list = [statistics.mean(current_io_time_list) for current_io_time_list in new_runtime_nested_list]
            current_memory_list = [statistics.mean(current_io_memory_list) for current_io_memory_list in new_memory_nested_list]

        execution_time = round(statistics.mean(current_time_list), 2)
        execution_memory = round(statistics.mean(current_memory_list), 2)

        return io_pass_rate, execution_time, execution_memory, evaluation_result_dict






# #####################################################################################################################
def postprocess_cot_generated_code_py_cpp(original_code_str, index=0, prefix='python'):
    code_str = original_code_str.strip()
    # print(f"\n\n### original_code_str, index: {index}. code_str: {code_str}")
    for prefix in [
        'python', 'Python', 'cpp', 'Cpp', 'c++', 'C++',
        ' python', ' Python', ' cpp', ' Cpp', ' c++', ' C++',
        'css', 'bash', 'markdown', 'perl', 'php', 'scss', 'yaml',
        'sh', 'sql', 'C', 'c', 'end_error_marker',
    ]:
        if (f"```{prefix}" in code_str) or (f"```\n{prefix}" in code_str):
            break
        if prefix == 'end_error_marker':
            print(f"\n\n### Prefix error: {prefix}, index: {index}. code_str[:20]: {code_str}")


    # Handle the case with one triple-backtick marker.
    if code_str.count("```") == 1:
        # print(f"\n\n### Error: one triple-backtick marker, index: {index}\noriginal_code_str:\n{original_code_str}")
        if f"```{prefix}" in code_str:
            code_str = code_str.split(f"```{prefix}")[1].strip()
        elif f"```\n{prefix}" in code_str:
            code_str = code_str.split(f"```\n{prefix}")[1].strip()
        elif f"```\n" in code_str:
            code_str = code_str.split(f"```\n")[1].split("```")[0].strip()
            print(f"\n\n### Single marker: check whether the first line has an error. index: {index}\ncode_str[:20]: {code_str[:20]}")
        else:
            print(f"\n\n### Need careful distinction. index: {index}\noriginal_code_str:\n{original_code_str}")
            code_str = code_str.split(f"```")[1].split("```")[0].strip()
        assert "```" not in code_str


    # Handle the case with two triple-backtick markers.
    elif code_str.count("```") == 2:
        if f"```{prefix}" in code_str:
            code_str = code_str.split(f"```{prefix}")[1].split("```")[0].strip()
        elif f"```\n{prefix}" in code_str:
            code_str = code_str.split(f"```\n{prefix}")[1].split("```")[0].strip()
        elif f"```\n" in code_str:
            code_str = code_str.split(f"```\n")[1].split("```")[0].strip()
            print(f"\n\n### Double marker: check whether the first line has an error. index: {index}\ncode_str[:20]: {code_str[:20]}")
        else:
            print(f"\n\n### Need careful distinction. index: {index}\noriginal_code_str:\n{original_code_str}")
            code_str = code_str.split(f"```")[1].split("```")[0].strip()
        assert "```" not in code_str


    # Handle the case with multiple triple-backtick markers.
    elif code_str.count("```") > 2:
        # print(f"\n\n### Error: multiple triple-backtick markers, index: {index}\noriginal_code_str:\n{original_code_str}")

        if f"```{prefix}" in code_str:
            code_str = code_str.split(f"```{prefix}")[1].split("```")[0].strip()
        if f"```\n{prefix}" in code_str:
            code_str = code_str.split(f"```\n{prefix}")[1].split("```")[0].strip()

        if "```" in code_str:
            code_str = code_str.split("```")[1]
            if (not code_str) or (code_str[0] != '\n'):
                print(f"\n\n### Error: multiple triple-backtick markers. index: {index}\noriginal_code_str:\n{original_code_str}")
                print(f"\n\n### After processing, code_str:\n{code_str}")
            code_str = code_str.strip()


        if "```" in code_str:
            print(f"\n\n### Error: multiple triple-backtick markers cannot be resolved. index: {index}\noriginal_code_str:\n{original_code_str}")
            print(f"\n\n### After processing, code_str:\n{code_str}")





    # --------------------------------------------------------------------------------------------------------------
    if code_str == "" or code_str == "nan":
        print(f"\n### code_str is empty. index: {index}\noriginal_code_str:\n{original_code_str}")
        code_str = 'pass'

    return code_str.strip()







# ################################################################################################################################################
if __name__ == '__main__':
    main()

