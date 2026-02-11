# -*- coding: utf-8 -*-
# print(f"### :\n{}")
# #####################################################################################################################🔖💡✅🟨❌

GLOBAL_TIMEOUT = 1  # Seconds

# utilities and code that handles actual execution of programs
import pathlib
import shlex
import subprocess
from typing import Dict, List, Tuple, Union
import time
import numpy as np
import resource
import logging
import psutil
import os
import sys
import traceback
import pdb
import glob

import argparse
import json
import statistics
import io

# disable logging from psutil
logging.getLogger("psutil").setLevel(logging.WARNING)

# disable logging from resource
logging.getLogger("resource").setLevel(logging.WARNING)

# disable logging from subprocess
logging.getLogger("subprocess").setLevel(logging.WARNING)

logging.basicConfig(level=logging.CRITICAL)

DEBUG=True



# #####################################################################################################################🔖💡✅🟨❌
'''from API__PIE_sandbox import create_cmd_process_and_run_python_code_io_unit_test_func'''
def create_cmd_process_and_run_python_code_io_unit_test_func(tested_code_path: str, io_file: str, ignore_count: int = 1, run_count_excluding_ignore: int = 26, programming_language: str = 'python', pie_index: str = '0'):
    """ io_file: Dictionary path """

    # Construct the command argument list. The first is the "python" executable, the second is the script to run "fun.py"
    # cmd_list = ["taskset", "--cpu-list", "0", "python", "API__PIE_sandbox__latest6.py", tested_code_str, io_file, str(ignore_count), str(run_count_excluding_ignore), str(programming_language), str(pie_index)]
    cmd_list = ["python", "API__PIE_sandbox__latest7.py", tested_code_path, io_file, str(ignore_count), str(run_count_excluding_ignore), str(programming_language), str(pie_index)]

    try:
        cmd_process_return_value = subprocess.run(cmd_list, 
                                                start_new_session=True, # Make the subprocess create a new session to avoid receiving signals from the parent process or sharing the control terminal
                                                capture_output=True,    # capture_output=True used to capture the output of .py
                                                text=True               # text=True makes the output a string instead of a byte stream
                                                )



        # 0: Usually indicates normal termination, successful completion (this is not mandatory, but a general convention).
        # Non-0: Indicates some error, exception, or custom exit status.
        subprocess_return_code = cmd_process_return_value.returncode

        # Represents all text written to standard error (stderr) by the subprocess during execution
        subprocess_stderr = cmd_process_return_value.stderr

        # --------------------------------------------------------------------
        # print("### === Subprocess Stderr ===")
        # print(cmd_process_return_value.stderr)

        # print("### === Subprocess Return Head ===")
        # print(cmd_process_return_value.stdout.strip())
        # print("### === Subprocess Return Tail ===")
        
        # Represents all text written to standard output (stdout) by the subprocess during execution
        # Restore to Python dictionary using json.loads
        io_test_result_dict = json.loads(cmd_process_return_value.stdout.strip())

        io_test_result_dict["Subprocess_Return_Code"] = subprocess_return_code
        io_test_result_dict["Subprocess_Stderr"] = subprocess_stderr

    except Exception as e:
        print(f"❌❌❌ Subprocess execution failed, likely due to code being too long! Error: {e}")
        return {
                "IO_Pass_Results": [0],
                "Time_Usage_List": [[1234567890]],
                "Time_Standard_Deviation": 0,
                "Time_List_Unit": 'milliseconds_ms = e-3 seconds',
                "Error_Type": ['Evaluation File Failed', str(e)],
                "Test_Code_IO_Output": [],
            }

    return io_test_result_dict




# #####################################################################################################################❌
def get_hyperparameters():
    parser = argparse.ArgumentParser(description="Command line execution.")
    parser.add_argument("tested_code_path")
    parser.add_argument("io_file")
    parser.add_argument("ignore_count", type=int)
    parser.add_argument("run_count_excluding_ignore", type=int)
    parser.add_argument("programming_language")
    parser.add_argument("pie_index")

    args = parser.parse_args()
    return args



# #####################################################################################################################❌
def main():
    args = get_hyperparameters()
    # print(pprint.pformat(vars(args)))

    # ---------------------------------------
    """ Read IO File """
    if type(args.io_file) == str:
        with open(args.io_file, 'r', encoding='UTF-8') as f:
            io_test_dict = eval(f.read())
    else:
        io_test_dict = args.io_file

    io_test_output_list = io_test_dict['outputs']

    io_test_count = len(io_test_output_list)
                
    assert (io_test_count > 0), f"{args.io_file} No IO test path!"

    # print(f"🔖🔖🔖 args: {args}")
    io_test_result_dict = run_code_io_unit_test_func( code_path = args.tested_code_path,
                                                        io_dict = io_test_dict,
                                                        ignore_count = args.ignore_count,
                                                        run_count_excluding_ignore = args.run_count_excluding_ignore,
                                                        io_ground_truth_list = io_test_output_list,    # type: ignore
                                                        programming_language = args.programming_language,  # "python" or "cpp"
                                                        pie_index = args.pie_index,
                                                        )
    # print(io_test_result_dict)
    print(json.dumps(io_test_result_dict))





# #####################################################################################################################🔖💡✅🟨
def run_code_io_unit_test_func(
    code_path: str,
    io_dict: str,
    ignore_count: int,
    run_count_excluding_ignore: int,
    timeout: int = GLOBAL_TIMEOUT,
    io_ground_truth_list: List[str] = None,    # type: ignore
    io_test_count: int = None,         # type: ignore
    designated_cpu_id: int = 18,                # which CPU to run the code on, counting begins from 1
    programming_language: str = "python",   # "cpp"
    pie_index: str = "0",
) -> Union[Tuple[float, float, float], Tuple[float, float, float, List[List[float]]], Dict]:
    """
    Run the given code on the input of the specified problem_id and return (avg_time, std_time, avg_acc, error_type).
    These inputs come from unit test data, where multiple files like {input,output}.{0, 1, 2}.txt exist.

    NOTE: Passing ground_truths is optional. If they are not passed, accuracy will not be calculated.
    """

    if io_test_count is None:
        io_test_count = len(io_ground_truth_list)

    if programming_language == "cpp":
        try:
            compiled_executable_path = compile_cpp_code_func(code_path, cflags="--std=c++20 -O3", pie_index=pie_index)
        except Exception as e:
            return {
                "IO_Pass_Results": [0],
                "Time_Usage_List": [[1234567890]],
                "Time_Standard_Deviation": 0,
                "Time_List_Unit": 'milliseconds_ms = e-3 seconds',
                "Error_Type": ['Compilation Error', str(e)],
                "Test_Code_IO_Output": [],
            }

    # ----------------------------------------------------------------------------------------------
    """ Create CMD Command: Example: taskset 00 python code.py"""
    if is_linux() and programming_language == "python": 
        cmd_str = ( f"taskset --cpu-list {designated_cpu_id} {programming_language} {code_path}" )
    elif is_linux() and programming_language == "cpp":
        cmd_str = (f"taskset --cpu-list {designated_cpu_id} {compiled_executable_path}" )
    else: 
        cmd_str = f"{programming_language} {code_path}"
    cmd_list = shlex.split(cmd_str)
    
    # ----------------------------------------------------------------------------------------------
    total_io_pass_single_list = []
    total_io_time_double_list = []
    total_error_type_list = []
    total_code_io_output_list = []
    for io_seq in range(io_test_count):
        io_input_text = io_dict['inputs'][io_seq]

        current_io_pass_list = []
        current_io_time_list = []
        error_type_list = []
        current_io_output_list = []

        for _current_run_count in range(ignore_count + run_count_excluding_ignore):
            try:
                test_code_io_output, error_type, current_code_exec_time = run_code_from_cmd_and_get_time_func(
                    cmd_list,
                    io_input_text=io_input_text,
                    timeout=timeout,
                )
                if error_type is None:
                    error_type = ''
                                
                # Timeout: Since we set a generous timeout, this should not happen
                if test_code_io_output == "Exceeded custom time":
                    current_io_pass_list = [0]
                    current_io_time_list = [1234567890]
                    error_type_list = ["Exceeded custom time: "+error_type]
                    current_io_output_list = ['Error IO Output']
                    break

                # Output is empty
                if test_code_io_output is None:
                    current_io_pass_list = [0]
                    current_io_time_list = [1234567890]
                    error_type_list = ["Output is empty: "+error_type]
                    current_io_output_list = ['Error IO Output']
                    break

                io_pass_rate = compare_output_with_ground_truth(test_code_io_output, io_ground_truth_list[io_seq])

                # IO Pass Rate < 0.9999
                if io_pass_rate < 0.9999:
                    current_io_pass_list = [io_pass_rate]
                    current_io_time_list = [current_code_exec_time * 1000]
                    error_type_list = ["IO Pass Rate < 0.9999: "+error_type]
                    current_io_output_list = ['Error IO Output']
                    break

                # Normal case
                if _current_run_count >= ignore_count:
                    current_io_time_list.append(current_code_exec_time * 1000)
                    if _current_run_count == ignore_count:
                        current_io_pass_list = [io_pass_rate]
                        error_type_list = [error_type]
                        current_io_output_list = [test_code_io_output]
                        
            except Exception as e:
                current_io_pass_list = [0]
                current_io_time_list = [1234567890]
                error_type_list = ["Evaluation program crashed:"+str(e)]
                current_io_output_list = ['Error IO Output']
                break
 
        # -----------------------------------------------------------------------------------
        assert len(current_io_pass_list) ==  1, f"current_io_pass_list length {len(current_io_pass_list)} is not 1 !"
        assert len(error_type_list) ==  1, f"error_type_list length {len(error_type_list)} is not 1 !"
        assert len(current_io_output_list) ==  1, f"current_io_output_list length {len(current_io_output_list)} is not 1 !"
        total_io_pass_single_list.extend(current_io_pass_list)
        total_io_time_double_list.append(current_io_time_list)
        total_error_type_list.extend(error_type_list)
        total_code_io_output_list.extend(current_io_output_list)

    # ============================================================================================================================================
    assert len(total_io_pass_single_list) == len(total_io_time_double_list) == len(total_error_type_list) == len(total_code_io_output_list), f"Length mismatch, IO Pass, IO Time, Error Type, IO Output: {len(total_io_pass_single_list)}, {len(total_io_time_double_list)}, {len(total_error_type_list)}, {len(total_code_io_output_list)}"

    if is_linux() and programming_language == "cpp" and os.path.exists(compiled_executable_path):
        os.remove(compiled_executable_path)

    rounded_total_run_time_double_list = [[round(x, 2) for x in sublist] for sublist in total_io_time_double_list]

    # -------------------------------------------------------------------------------------------------------------
    return {
        "IO_Pass_Results": total_io_pass_single_list,
        "Time_Usage_List": rounded_total_run_time_double_list,
        "Time_Standard_Deviation": round(statistics.pstdev([x for sublist in total_io_time_double_list for x in sublist]), 4),
        "Time_List_Unit": 'milliseconds_ms = e-3 seconds',
        "Error_Type": total_error_type_list,
        "Test_Code_IO_Output": total_code_io_output_list,
    }





# #####################################################################################################################🔖💡✅🟨
def is_linux():
    from sys import platform
    if platform == "linux" or platform == "linux2":
        return True
    else: 
        return False




# #####################################################################################################################🔖💡✅🟨
# Maximal virtual memory for subprocesses (in bytes).
MAX_VIRTUAL_MEMORY = 10 * 1024 * 1024 * 50  # 500 MB

# from https://gist.github.com/s3rvac/f97d6cbdfdb15c0a32e7e941f7f4a3fa
def limit_virtual_memory():
    # The tuple below is of the form (soft limit, hard limit). Limit only the soft part
    # so that the limit can be increased later (setting also the hard limit would prevent this).
    # When the limit cannot be changed, setrlimit() raises ValueError.
    if is_linux():
        resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, MAX_VIRTUAL_MEMORY * 10))
    else: 
        pass
    

# #####################################################################################################################🔖💡✅🟨
def run_code_from_cmd_and_get_time_func(cmd_list, io_input_text: str, timeout: int = GLOBAL_TIMEOUT) -> Union[str, None]:
    def _kill(proc_pid):
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()

    try:
        proc = subprocess.Popen(
            cmd_list,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=limit_virtual_memory,
        )
        
        start_time = time.time()
        output, error_type = proc.communicate(input=io_input_text.encode('utf-8'), timeout=timeout)

        current_code_execution_time = time.time() - start_time
        
        return output.decode("utf-8").strip(), error_type.decode("utf-8").strip(), current_code_execution_time
    except subprocess.TimeoutExpired:
        _kill(proc.pid)  # type: ignore
        return "Exceeded custom time", "subprocess.TimeoutExpired System Timeout", 1234567890


# ####################################################################################################################################################
def compare_output_with_ground_truth(test_io_output: str, ground_truth_io_output: str) -> float:
    """
    Compare code output with ground truth.
    """
    correct_count = 0
    output_truth_lines = test_io_output.strip().splitlines()
    ground_truth_lines = ground_truth_io_output.strip().splitlines()
    for model_output, io_ground_truth in zip(output_truth_lines, ground_truth_lines):
        is_correct = model_output == io_ground_truth

        # -------------------- Irrelevant Whitespace -----------------------#
        if not is_correct:
            model_output = model_output.strip()
            io_ground_truth = io_ground_truth.strip()
            is_correct = model_output == io_ground_truth

        # -------------------- 1 vs 0.9999999 Case -----------------------#
        if not is_correct:
            try:
                model_output = float(model_output)
                io_ground_truth = float(io_ground_truth)
                is_correct = abs(model_output - io_ground_truth) < 1e-3
            except:
                pass

        # -------------------- Yes vs YES Case -----------------------#
        if not is_correct:
            try:
                is_correct = model_output.lower() == io_ground_truth.lower()
            except:
                pass

        correct_count += int(is_correct)

    if correct_count == 0:
        return correct_count
    return correct_count / len(ground_truth_lines)




# #####################################################################################################################🔖💡✅🟨
def write_io_test_input_func(inputs=["10000", "1000000"]):

    # Write inputs to a temporary directory.
    temp_dir_name = create_temp_directory_func()

    # Create files containing ground truths, named inputs.{i}.txt
    for i, input_txt in enumerate(inputs):
        with open(f"{temp_dir_name}/input.{i}.txt", "w") as input_file:
            print(f"Wrote input # {i} to {input_file.name}")
            input_file.write(input_txt)
            
    io_test_output_values = [str(sum(range(int(i) + 1))) for i in inputs]

    return io_test_output_values, temp_dir_name


# #####################################################################################################################🔖💡✅🟨
def create_temp_directory_func():
    import uuid
    temp_dir_name = f"/tmp/{uuid.uuid4()}"
    pathlib.Path(temp_dir_name).mkdir(parents=True, exist_ok=True)
    return temp_dir_name


# #####################################################################################################################🔖💡✅🟨
def compile_cpp_code_func(code_path: str, compiled_executable_path: str = None, cflags: str = "", pie_index: str = "0") -> str:
    """_summary_

    Args:
        code_path (str): _description_
        output_path (str, optional): _description_
        cflags (str, optional): _description_
    
    Returns:
        str: _description_
    """
    if compiled_executable_path is None:
        compiled_executable_path = os.path.join(os.path.dirname(code_path), f"./Cpp_{pie_index}.out")
        # print(f"### Constructed key output path: {compiled_executable_path}")
    cmd = ["/usr/bin/g++-13", code_path, "-o", compiled_executable_path] + shlex.split(cflags.replace('"', ""). replace("'", ""))

    # print(f"### cmd compilation command: {cmd}")

    p = subprocess.run(cmd, capture_output=True)
    
    if p.returncode != 0:
        raise Exception(f"Error compiling code: {code_path} with command: {' '.join(cmd)}, return code: {p.returncode}, stderr: {p.stderr.decode('utf-8')}")
    
    return compiled_executable_path




# #####################################################################################################################🔖💡✅🟨
def run_c_code_io_unit_test_func(
    code_path: str,
    unit_test_data_basepath: str,
    ignore_count: int,
    run_count_excluding_ignore: int,
    timeout: int,
    io_ground_truth_list: List[str] = None,  # type: ignore
    io_test_count: int = None,  # type: ignore
    cpu_usage_count: int = 1,  # which CPU to run the code on, counting begins from 1
    return_per_trial_times: bool = False,
    python_bin: str = "python", # unused
    return_dict: bool = False,
    remove_code_after_run: bool = True, 
    debug_stderr = sys.stderr, # temporary for debugging purposes
    cflags: str = "--std=c++17 -O1",
    return_if_acc_below: float = 0.0,
) -> Union[Tuple[float, float, float], Tuple[float, float, float, List[List[float]]], Dict]:
    """
    Run the given code on the input of the specified problem_id and return (avg_time, std_time, avg_acc).
    These inputs come from unit test data, where multiple files like {input,output}.{0, 1, 2}.txt exist.

    NOTE: Passing ground_truths parameter is optional. If not passed, accuracy will not be calculated.
    """
    try: 
        binary_output_path = compile_cpp_code_func(code_path, cflags=cflags)
    except Exception as e:
        logging.warning(f"Error: {e}")
        return (np.nan, np.nan, 0)
        
    if io_test_count is None:
        io_test_count = len(io_ground_truth_list)

    total_run_time_double_list = []
    total_io_pass_list = []
    for io_seq in range(io_test_count):
        if is_linux(): 
            cmd = (f"taskset --cpu-list {cpu_usage_count} {binary_output_path}" )
        else: 
            cmd = f"{binary_output_path}"
        subprocess_args = shlex.split(cmd)
        input_file_path = f"{unit_test_data_basepath}/input.{io_seq}.txt"
        _per_trial_times = []
        for _current_run_count in range(run_count_excluding_ignore):
            try:
                time_start = time.time()
                output, error_type = run_code_from_cmd_and_get_time_func(
                    subprocess_args,
                    io_input_text=input_file_path,
                    timeout=timeout,
                )
                time_taken = time.time() - time_start
                _per_trial_times.append(time_taken)
                if output is None:
                    if remove_code_after_run: 
                        os.remove(binary_output_path)
                    return (np.nan, np.nan, 0)
                    # timeout: since we have a generous timeout, this should not happen

                if _current_run_count >= ignore_count:
                    total_run_time_double_list.append(time_taken * 1000)
                    if io_ground_truth_list is not None:
                        io_pass_rate = compare_output_with_ground_truth(output, io_ground_truth_list[io_seq])
                        if io_pass_rate < return_if_acc_below:
                            if remove_code_after_run: 
                                os.remove(binary_output_path)
                            logging.info(f"Accuracy {io_pass_rate} below {return_if_acc_below}. Returning.")
                            return (np.nan, np.nan, 0)
                        total_io_pass_list.append(io_pass_rate)

            except Exception as e:
                logging.warning("Error", e)
                # no point in repeating the test for this problem. If something went wrong, it will go wrong again
                return (np.nan, np.nan, 0)


    total_run_time_double_list, total_io_pass_list = np.array(total_run_time_double_list), np.array(total_io_pass_list)

    if return_dict:
        return {
            "avg_time": np.mean(total_run_time_double_list),
            "std_time": np.std(total_run_time_double_list),
            "avg_acc": np.mean(total_io_pass_list),
        }
    else:
        return np.mean(total_run_time_double_list), np.std(total_run_time_double_list), np.mean(total_io_pass_list)  # type: ignore


# #####################################################################################################################🔖💡✅🟨
def test_cpp_case():
    import shutil
    from pprint import pprint
    slow_sum_code_path = "src/codenet_eval/cpp_examples/slow_num.cpp"
    fast_num_code_path = "src/codenet_eval/cpp_examples/fast_num.cpp"
    fast_but_wrong_code_path = "src/codenet_eval/cpp_examples/fast_but_wrong.cpp"
    test_cases = {
        "slow": slow_sum_code_path,
        "fast": fast_num_code_path,
        "fast_but_wrong": fast_but_wrong_code_path
    }
    ground_truths, temp_dir_name = write_io_test_input_func()
    results = {code_type: {} for code_type in test_cases}
    for (code_type, code_pth) in test_cases.items():
        code_type_results = run_c_code_io_unit_test_func(  # type: ignore
            code_path=code_pth,
            unit_test_data_basepath=temp_dir_name,
            run_count_excluding_ignore=10,
            ignore_count=2,
            timeout=10,
            io_ground_truth_list=ground_truths,
            cpu_usage_count=2,
            return_dict = True
        )
        results[code_type].update(code_type_results)  # type: ignore
    
    assert results["slow"]["avg_time"] > results["fast"]["avg_time"]
    assert results["fast"]["avg_acc"] == 1.0
    assert results["slow"]["avg_acc"] == 1.0
    assert results["fast_but_wrong"]["avg_acc"] == 0.0
    shutil.rmtree(temp_dir_name)
    print("Test passed! Results: ")
    pprint(results)
    
    


# ########################################################################################################################################################
def test_case_main_func(): 
    test_cpp_case()
    

# #####################################################################################################################🔖💡✅🟨
if __name__ == "__main__":
    main()