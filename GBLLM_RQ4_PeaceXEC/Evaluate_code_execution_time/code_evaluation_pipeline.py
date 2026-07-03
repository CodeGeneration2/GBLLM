from git_checkout import switch_repo_to_version
from function_replacer import replace_function_with_llm_generated_code, replace_target_file_function_body_with_source
from test_performance_extractor import run_test_and_get_output, extract_cpu_instruction_count, extract_execution_time, extract_memory_usage


# #####################################################################################################################🔖💡🟨✓✗
def evaluate_code_performance_after_replacement(total_eval_data_dict, is_slow_code=False, ignore_count=0, run_count_excluding_ignore=1):
    """
    Evaluate the performance of the function after code replacement.

    Args:
        total_eval_data_dict (dict): Dictionary containing information required for evaluation.
            Keys include:
                - repo_path: Path to the repository.
                - file_path: Path to the file containing the function.
                - sha: Git commit SHA to switch to.
                - function_name: Name of the function to replace.
                - class_name: Class name if the function is a method.
                - venv_path: Path to the Python virtual environment.
                - test_cmd: Test command to run.
                - after_code: New code for the function.

    Returns:
        tuple: Tuple containing CPU instruction count and memory usage.
    """
    # Extract data from input dictionary
    repo_root_dir = "../repo_python/" + total_eval_data_dict["repo_path"]
    version_id = total_eval_data_dict["sha"]
    file_path = total_eval_data_dict["target_file"]
    target_class_name = total_eval_data_dict["target_class"]
    target_func_name = total_eval_data_dict["target_func"]
    venv_path = "../../venv_python/" + total_eval_data_dict["venv_path"]
    test_cmd = total_eval_data_dict["test_cmd"]

    # Switch repository to specified version
    switch_repo_to_version(repo_root_dir, version_id)

    # Replace function with LLM generated code
    if is_slow_code == False:
        new_function_body = total_eval_data_dict["after_code"]
        replace_result = replace_target_file_function_body_with_source(file_path, target_func_name, new_function_body, target_class_name)
        if replace_result == False:
            return [0], [1234567890], [1234567890], [1234567890]


    io_pass_rate_list = []
    cpu_instruction_count_list = []
    execution_time_list = []
    memory_usage_list = []

    for _ in range(ignore_count):
        # Run test and get output
        test_output = run_test_and_get_output(repo_root_dir, venv_path, test_cmd)
        if test_output is None:
            return [0], [1234567890], [1234567890], [1234567890]
        # print("### Ignoring test output.")
        # print(test_output)
    for _ in range(run_count_excluding_ignore):
        # Run test and get output
        test_output = run_test_and_get_output(repo_root_dir, venv_path, test_cmd)
        if test_output is None:
            return [0], [1234567890], [1234567890], [1234567890]
        # print("### Test output.")
        # print(test_output)
        # Extract CPU instruction count and memory usage from output
        cpu_instruction_count = extract_cpu_instruction_count(test_output)
        execution_time = extract_execution_time(test_output)
        memory_usage = extract_memory_usage(test_output)
        io_pass_rate_list.append(1)
        cpu_instruction_count_list.append(cpu_instruction_count)
        execution_time_list.append(1000*1000*execution_time)
        memory_usage_list.append(memory_usage)

    assert len(cpu_instruction_count_list) == len(execution_time_list) == len(memory_usage_list) == run_count_excluding_ignore, "Extracted performance data length mismatch"

    return io_pass_rate_list, cpu_instruction_count_list, execution_time_list, memory_usage_list



# #####################################################################################################################🔖💡🟨 ✅✅✅ ❌❌❌
new_function_body = r'''
def lines_with_leading_tabs_expanded(s: str) ->List[str]:
    """
    Splits string into lines and expands only leading tabs (following the normal
    Python rules)
    """
    lines = []

    return lines
'''
def test_run_fast_code():
    """
    Main function to execute evaluation.
    """
    # Sample data for evaluation
    total_eval_data_dict = {
        "owner":"optuna",
        "reponame":"optuna",
        "target_file":"optuna/_hypervolume/hssp.py",
        "target_class": None,
        "target_func":"_solve_hssp",
        "desc":"",
        "sha":"e75b8763492d5ce4ce2c231044eed8d04498fd25",
        "repo_path":"optuna",
        "venv_path":"optuna_optuna",
        "test_cmd":"pytest -s tests/hypervolume_tests/test_hssp.py::test_solve_hssp",
        "install_virtual_env":"Success: "
    }

    # Execute evaluation
    slow_io_pass_rate_list, slow_cpu_instruction_count_list, slow_execution_time_list, slow_memory_usage_list = evaluate_code_performance_after_replacement(total_eval_data_dict, is_slow_code=True, ignore_count=1, run_count_excluding_ignore=3)

    fast_io_pass_rate_list, fast_cpu_instruction_count_list, fast_execution_time_list, fast_memory_usage_list = evaluate_code_performance_after_replacement(total_eval_data_dict, is_slow_code=False, ignore_count=1, run_count_excluding_ignore=3)


    slow_io_pass_rate = round(sum(slow_io_pass_rate_list) / len(slow_io_pass_rate_list), 2)
    slow_cpu_instruction_count = round(sum(slow_cpu_instruction_count_list) / len(slow_cpu_instruction_count_list))
    slow_execution_time = round((sum(slow_execution_time_list) / len(slow_execution_time_list)), 2)
    slow_memory_usage = round(sum(slow_memory_usage_list) / len(slow_memory_usage_list), 2)

    fast_io_pass_rate = round(sum(fast_io_pass_rate_list) / len(fast_io_pass_rate_list), 2)
    fast_cpu_instruction_count = round(sum(fast_cpu_instruction_count_list) / len(fast_cpu_instruction_count_list))
    fast_execution_time = round((sum(fast_execution_time_list) / len(fast_execution_time_list)), 2)
    fast_memory_usage = round(sum(fast_memory_usage_list) / len(fast_memory_usage_list), 2)

    # Print results
    print(f"\n### IO Pass Rate: {slow_io_pass_rate} (Slow), {fast_io_pass_rate} (Fast)")
    print(f"### CPU Instructions: {slow_cpu_instruction_count} (Slow), {fast_cpu_instruction_count} (Fast)")
    print(f"### Execution Time: {slow_execution_time} microseconds (us) (Slow), {fast_execution_time} microseconds (us) (Fast)")
    print(f"### Memory Usage: {slow_memory_usage} MB (Slow), {fast_memory_usage} MB (Fast)")




# #####################################################################################################################🔖💡✅🟨
def main():
    """
    Main function to execute evaluation.
    """
    # Sample data for evaluation
    total_eval_data_dict = {
        "owner":"optuna",
        "reponame":"optuna",
        "target_file":"optuna/_hypervolume/hssp.py",
        "target_class": None,
        "target_func":"_solve_hssp",
        "desc":"",
        "sha":"e75b8763492d5ce4ce2c231044eed8d04498fd25",
        "repo_path":"optuna",
        "venv_path":"optuna_optuna",
        "test_cmd":"pytest -s tests/hypervolume_tests/test_hssp.py::test_solve_hssp",
        "install_virtual_env":"Success: "
    }
    # python -m pytest -s --override-ini="addopts=" tests/test_core.py::test_magphase_zero

    # Execute evaluation
    slow_io_pass_rate_list, slow_cpu_instruction_count_list, slow_execution_time_list, slow_memory_usage_list = evaluate_code_performance_after_replacement(total_eval_data_dict, is_slow_code=True, ignore_count=0, run_count_excluding_ignore=1)


    slow_io_pass_rate = round(sum(slow_io_pass_rate_list) / len(slow_io_pass_rate_list), 2)
    slow_cpu_instruction_count = round(sum(slow_cpu_instruction_count_list) / len(slow_cpu_instruction_count_list))
    slow_execution_time = round((sum(slow_execution_time_list) / len(slow_execution_time_list)), 2)
    slow_memory_usage = round(sum(slow_memory_usage_list) / len(slow_memory_usage_list), 2)


    # Print results
    print(f"\n### IO Pass Rate: {slow_io_pass_rate} (Slow)" )
    print(f"### CPU Instructions: {slow_cpu_instruction_count} (Slow)")
    print(f"### Execution Time: {slow_execution_time} microseconds (us) (Slow)")
    print(f"### Memory Usage: {slow_memory_usage} MB (Slow)")




# #####################################################################################################################🔖💡✅🟨
"""
Code Functionality Summary

    The purpose of this code is to evaluate the performance of a function after code replacement. It assesses the performance impact, primarily CPU instruction count and memory usage, of replacing a function by executing the following steps. It integrates multiple functional modules, including Git operations, function replacement, test execution, and performance data extraction.

    
Key Features Overview:

    Switch to a specified Git commit version (SHA)

    Replace function implementation in the file

    Run test commands to evaluate function performance

    Extract performance data from test results: CPU instruction count and memory usage

"""
if __name__ == "__main__":
    main()