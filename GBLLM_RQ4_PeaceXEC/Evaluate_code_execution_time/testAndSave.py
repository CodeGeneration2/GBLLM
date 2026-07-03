import json
import os
import subprocess
import re
import ast
import statistics


# #####################################################################################################################🔖💡✅🟨
def load_json_file(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON data.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Unable to decode JSON from {file_path}.")
        return {}


# #####################################################################################################################🔖💡✅🟨
def save_json_file(data, file_path):
    """
    Save data as a JSON file.

    Args:
        data (dict): The data to save.
        file_path (str): The path for the output JSON file.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Results successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving results to {file_path}: {e}")


# #####################################################################################################################🔖💡✅🟨
def replace_function_in_file(file_path, optimized_function_code, function_name):
    """
    Replace the original function in the specified file with optimized code,
    maintaining the correct indentation within classes or nested structures.

    Args:
        file_path (str): Path to the file containing the function.
        optimized_function_code (str): The optimized function code.
        function_name (str): The name of the function to replace.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        tree = ast.parse(content)
        start_line = None
        end_line = None

        class FunctionFinder(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                if node.name == function_name:
                    nonlocal start_line, end_line
                    start_line = node.lineno
                    end_line = node.end_lineno

        FunctionFinder().visit(tree)

        if start_line and end_line:
            content_lines = content.splitlines()
            # Calculate original indentation
            original_indent = len(content_lines[start_line - 1]) - len(content_lines[start_line - 1].lstrip())
            optimized_lines = optimized_function_code.splitlines()
            # Apply indentation to new code
            adjusted_optimized_lines = [(' ' * original_indent) + line if line else line for line in optimized_lines]
            
            new_content_lines = (
                content_lines[:start_line - 1] +
                adjusted_optimized_lines +
                content_lines[end_line:]
            )

            with open(file_path, 'w', encoding='utf-8') as file:
                file.write("\n".join(new_content_lines) + "\n")
            print(f"Function '{function_name}' successfully replaced in {file_path}")
        else:
            print(f"Function '{function_name}' not found in {file_path}")
    except Exception as e:
        print(f"Error replacing function in file {file_path}: {e}")


# #####################################################################################################################🔖💡✅🟨
def extract_cpu_instr(output):
    """
    Extract CPU instruction count from pytest output.

    Args:
        output (str): Output from the pytest command.

    Returns:
        int: CPU instruction count, or -1 if not found.
    """
    match = re.search(r'instruction_count=(\d+)', output)
    return int(match.group(1)) if match else -1


# #####################################################################################################################🔖💡✅🟨
def extract_mem_usage(output):
    """
    Extract memory usage from pytest output.

    Args:
        output (str): Output from the pytest command.

    Returns:
        float: Memory usage (in MB), or -1 if not found.
    """
    match = re.search(r'Memory usage \(MB\): ([\d.]+)', output)
    return float(match.group(1)) if match else -1


# #####################################################################################################################🔖💡✅🟨
def run_single_test(repo_info, venv_path, test_cmd, init_workdir):
    """
    Run a single test and extract CPU instruction count and memory usage.

    Args:
        repo_info (dict): Information about the repository.
        venv_path (str): Path to the virtual environment.
        test_cmd (str): The test command to run.
        init_workdir (str): The initial working directory.

    Returns:
        tuple: A tuple containing CPU instruction count, memory usage, and acceptance status (1 for success, 0 for failure).
    """
    repo_name = os.path.join(init_workdir, "test_repo_python", repo_info['repo_path'])
    target_file = os.path.join(repo_name, repo_info['target_file'])
    target_func = repo_info['target_func']
    sha = repo_info['sha']

    try:
        os.chdir(repo_name)
        subprocess.run(f"git checkout {sha} -f", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Repository '{repo_name}' reset to SHA '{sha}'")
    except subprocess.CalledProcessError as e:
        print(f"Failed to reset repository '{repo_name}' to SHA '{sha}': {e.stderr}")
        return -1, -1, 0

    full_command = f"export PYTHONPATH=$PYTHONPATH:$(pwd) && bash -c 'source {venv_path}/bin/activate && {test_cmd}'"
    try:
        result = subprocess.run(
            full_command,
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        cpu_instr = extract_cpu_instr(result.stdout)
        mem_usage = extract_mem_usage(result.stdout)
        return cpu_instr, mem_usage, 1
    except subprocess.CalledProcessError as e:
        print(f"Test failed for {repo_name}:\n{e.stderr}")
        return -1, -1, 0


# #####################################################################################################################🔖💡✅🟨
def run_tests(repo_data, optimized_functions, init_workdir, num_runs=12):
    """
    Iterate through repository configurations, run tests, and collect results.

    Args:
        repo_data (dict): Repository configuration data.
        optimized_functions (dict): Optimized function codes.
        init_workdir (str): Initial working directory.
        num_runs (int): Number of test runs.

    Returns:
        dict: Test results.
    """
    results = {}
    for repo_key in repo_data:
        repositories = repo_data[repo_key]
        for repo in repositories:
            sha = repo['sha']
            venv_path = os.path.join(init_workdir, "venv_python", repo['venv_path'])
            test_cmd = repo['test_cmd']
            target_file = os.path.join(os.path.join(init_workdir, "test_repo_python", repo['repo_path']), repo['target_file'])
            target_func = repo['target_func']

            if sha in optimized_functions:
                optimized_function_code = optimized_functions[sha]
                replace_function_in_file(target_file, optimized_function_code, target_func)
                print(f"Replaced function '{target_func}' in '{target_file}'.")

            cpu_instr_list = []
            mem_usage_list = []
            accepted = 1
            error_message = ""

            cpu_instr, mem_usage, initial_accepted = run_single_test(repo, venv_path, test_cmd, init_workdir)
            if initial_accepted:
                if cpu_instr != -1:
                    cpu_instr_list.append(cpu_instr)
                if mem_usage != -1:
                    mem_usage_list.append(mem_usage)
            else:
                accepted = 0
                error_message = f"Initial test failed"

            if accepted:
                for _ in range(num_runs - 1):
                    cpu_instr, mem_usage, run_accepted = run_single_test(repo, venv_path, test_cmd, init_workdir)
                    if run_accepted:
                        if cpu_instr != -1:
                            cpu_instr_list.append(cpu_instr)
                        if mem_usage != -1:
                            mem_usage_list.append(mem_usage)
                    else:
                        accepted = 0
                        error_message = f"Failed during repeat testing"
                        break

            if len(cpu_instr_list) > 2:
                # Calculate mean excluding the min and max values (trimming outliers)
                cpu_instr_avg = statistics.mean(sorted(cpu_instr_list)[1:-1])
            else:
                cpu_instr_avg = -1

            if len(mem_usage_list) > 2:
                # Calculate mean excluding the min and max values
                mem_usage_avg = statistics.mean(sorted(mem_usage_list)[1:-1])
            else:
                mem_usage_avg = -1

            test_result = {
                "sha": sha,
                "test_cmd": test_cmd,
                "cpu_instr_avg": cpu_instr_avg,
                "mem_usage_avg": mem_usage_avg,
                "accepted": accepted,
                "error_message": error_message
            }

            if repo['reponame'] not in results:
                results[repo['reponame']] = []
            results[repo['reponame']].append(test_result)

    return results


# #####################################################################################################################🔖💡✅🟨
def main():
    """
    Main function to run tests and save results.
    """
    init_workdir = os.getcwd()
    repo_config_file = 'repo_config_file_path_placeholder'
    optimized_functions_file = 'optimized_functions_file_path_placeholder'
    output_results_file = 'output_results_file_path_placeholder'

    repo_data = load_json_file(repo_config_file)
    optimized_functions = load_json_file(optimized_functions_file)
    results = run_tests(repo_data, optimized_functions, init_workdir)
    save_json_file(results, output_results_file)


# #####################################################################################################################🔖💡✅🟨
if __name__ == "__main__":
    main()