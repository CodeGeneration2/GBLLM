import json
from code_evaluation_pipeline import evaluate_code_performance_after_replacement


# #####################################################################################################################🔖💡✅🟨
def load_json_file(file_path):
    """
    Load JSON data from a file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"### Error: File {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"### Error: Unable to decode JSON from {file_path}.")
        return {}


# #####################################################################################################################🔖💡✅🟨
def save_json_file(data_dict, file_path):
    """
    Save data to a file in JSON format.

    Args:
        data_dict (dict): The data to save.
        file_path (str): The output JSON file path.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False, indent=4)
        print("Evaluation results saved successfully.")
    except Exception as e:
        print(f"Error saving evaluation results to {file_path}: {e}")


# #####################################################################################################################🔖💡✅🟨
def preprocess_evaluation_data(modification, repo_info_item, version_id, venv_path_prefix):
    """
    Prepare data to be passed to the evaluate_code_performance_after_replacement function.

    Args:
        modification (dict): Dictionary containing modification details.
        repo_info_item (dict): Dictionary containing repository information.
        version_id (str): The commit SHA.
        venv_path_prefix (str): The prefix for the virtual environment path.

    Returns:
        dict: Prepared evaluation data.
    """
    modification["repo_path"] = modification["repo_path"].replace("DOCKER_REPOEXEC_PATH_PLACEHOLDER", "LOCAL_REPOEXEC_PATH_PLACEHOLDER")
    modification["file_path"] = modification["file_path"].replace("DOCKER_REPOEXEC_PATH_PLACEHOLDER", "LOCAL_REPOEXEC_PATH_PLACEHOLDER")
    return {
        "repo_path": modification["repo_path"],
        "venv_path": f"{venv_path_prefix}{repo_info_item['venv_path']}",
        "test_cmd": repo_info_item["test_cmd"],
        "sha": version_id,
        "function_name": modification["function_name"],
        "class_name": modification["class_name"],
        "file_path": modification["file_path"],
        "after_code": modification["after"]
    }


# #####################################################################################################################🔖💡✅🟨
def evaluate_modifications(total_repo_info_dict, modifications_file_data, venv_path_prefix):
    """
    Evaluate modifications in each repository.

    Args:
        total_repo_info_dict (dict): Repository information loaded from JSON.
        modifications_file_data (dict): Modification data loaded from JSON.
        venv_path_prefix (str): The prefix for the virtual environment path.

    Returns:
        dict: Evaluation results.
    """
    evaluation_results_dict = {}
    for repo_name, sha_results in modifications_file_data.items():
        repo_evaluation_results = {}
        project_list = total_repo_info_dict.get(repo_name, [])
        for version_id, modifications in sha_results.items():
            try:
                # "版本号" corresponds to "version_id" or "sha" in the translated context, 
                # assuming the source JSON key is also being translated or matched correctly.
                # If the source JSON key is still Chinese "版本号", keep it as repo_info_item["版本号"].
                # Assuming here the source data keys are also consistent with English or original structure.
                # To be safe with the prompt's instruction to translate Chinese characters:
                repo_info_item = next((item for item in project_list if item["version_id"] == version_id), None)
                
                if repo_info_item:
                    target_func_name = repo_info_item["target_func"]
                    for modification in modifications:
                        if modification["function_name"] != target_func_name:
                            continue
                        total_eval_data = preprocess_evaluation_data(modification, repo_info_item, version_id, venv_path_prefix)
                        cpu_instr, execution_time, mem_usage = evaluate_code_performance_after_replacement(total_eval_data)
                        repo_evaluation_results[version_id] = {
                            "cpu_instr": cpu_instr,
                            "mem_usage": mem_usage
                        }
                        break
            except Exception as e:
                print(f"Error evaluating {repo_name} at SHA {version_id}: {e}")
        evaluation_results_dict[repo_name] = repo_evaluation_results

    return evaluation_results_dict


# #####################################################################################################################🔖💡✅🟨
def main():
    """
    Main function to execute the evaluation process.
    """
    # Load repository information and modification data
    repo_info_file_path = "REPO_INFO_FILE_PATH_PLACEHOLDER"
    modifications_file_path = "MODIFICATIONS_FILE_PATH_PLACEHOLDER"
    total_repo_info_dict = load_json_file(repo_info_file_path)
    modifications_file_data = load_json_file(modifications_file_path)

    # Prefix for virtual environment path
    venv_path_prefix = "VENV_PATH_PREFIX_PLACEHOLDER"

    # Evaluate modifications
    evaluation_results_dict = evaluate_modifications(total_repo_info_dict, modifications_file_data, venv_path_prefix)

    # Save evaluation results
    output_file_path = "OUTPUT_FILE_PATH_PLACEHOLDER"
    save_json_file(evaluation_results_dict, output_file_path)


# #####################################################################################################################🔖💡✅🟨
if __name__ == "__main__":
    main()