import subprocess
import os
import re


# #####################################################################################################################🔖💡🟨✓✗
DEBUG = False


# #####################################################################################################################🔖💡✅🟨
def extract_cpu_instruction_count(output):
    """
    Extract CPU instruction count from pytest output.
    Returns -1 if no match is found.

    Args:
        output (str): Output from the test command.

    Returns:
        int: CPU instruction count, or -1 if not found.
    """
    match = re.search(r'instruction_count=(\d+)', output)
    return int(match.group(1)) if match else -1


# #####################################################################################################################🔖💡✅🟨
def extract_execution_time(output):
    """
    Extract execution time from pytest output.
    Returns -1 if no match is found.
    """
    match = re.search(r'Elapsed\s*time:\s*([\d.]+)\s*seconds', output)
    return float(match.group(1)) if match else -1



# #####################################################################################################################🔖💡✅🟨
def extract_memory_usage(output):
    """
    Extract memory usage from pytest output.
    Returns -1 if no match is found.

    Args:
        output (str): Output from the test command.

    Returns:
        float: Memory usage (in MB), or -1 if not found.
    """
    match = re.search(r'Memory usage \(MB\): ([\d.]+)', output)
    return float(match.group(1)) if match else -1


# #####################################################################################################################🔖💡✅🟨
def run_test_and_get_output(repo_root_dir, venv_path, test_cmd):
    """
    Switch repository to the specified version and run the test command
    to verify if the modified code is effective.

    Args:
        repo_root_dir (str): Root directory of the repository.
        venv_path (str): Path to the Python virtual environment.
        test_cmd (str): The test command to execute.

    Returns:
        str or None: Standard output of the test command if successful;
                     None if an error occurs.
    """
    try:
        # Get current working directory, switch to the specified repository directory
        if not os.getcwd().endswith(repo_root_dir.replace('../', '')):
            os.chdir(repo_root_dir)

        # Build the full command: activate virtual environment and run test command
        full_command = f"export PYTHONPATH=$PYTHONPATH:$(pwd) && bash -c 'source {venv_path}/bin/activate && {test_cmd}'"

        # test_command = "bash -c 'source ../../venv_python/scipy_scipy/bin/activate && pip install -e ../repo_python/scipy --no-build-isolation && cd ../repo_python/scipy && pytest scipy/optimize/tests/test_isotonic_regression.py -s'"

        if DEBUG:
            print(f"\n🔖🔖🔖 Running test command: {full_command}\n")

        # Run command and capture output
        result = subprocess.run(
            full_command,
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30  # Set timeout to 30 seconds
        )

        # Return standard output
        return result.stdout
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌❌❌ Error during test execution, Repository: {repo_root_dir}")
        if DEBUG:
            print(f"\n❌❌❌ Error during test execution: {e}")
            print(f"\n❌❌❌ stdout: {e.stdout}")
            print(f"\n❌❌❌ stderr: {e.stderr}")
        return None
    
    except Exception as e:
        print(f"\n❌❌❌ Unexpected error occurred: {e}")
        return None


# #####################################################################################################################🔖💡✅🟨
def main():
    """
    Main function to run tests and extract performance metrics.
    """
    # Use os.getcwd() to get the current working directory
    print(f"### Current working directory: {os.getcwd()}")

    # Replace these paths with actual paths before running
    repo_root_dir = "repo_python/black"
    venv_path = "../../venv_python/psf_black"
    test_cmd = "pytest -s tests/test_black.py::BlackTestCase::test_lines_with_leading_tabs_expanded"


    # ----------------------------------------------------------------------
    output = run_test_and_get_output(repo_root_dir, venv_path, test_cmd)
    if output:
        print("### Code validation passed:")
    else:
        print("### Error occurred during test execution.")

    # print("### Raw test output below:")
    # print(output)

    # Extract CPU instructions and memory usage
    cpu_instr = extract_cpu_instruction_count(output)
    exec_time = extract_execution_time(output)
    mem_usage = extract_memory_usage(output)
    print(f"### CPU Instructions: {cpu_instr}")
    print(f"### Execution Time: {1000*1000*exec_time} microseconds (us)")
    print(f"### Memory Usage: {mem_usage} MB")


# #####################################################################################################################🔖💡✅🟨
if __name__ == "__main__":
    main()