import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 only.

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import logging

from API__Single_Generation import deepseek_dual_platform  # Used to call the DeepSeek API wrapper.
from API__Single_Generation import codelama_deepinfra
from API__Single_Generation import gemini_official
from API__Single_Generation import gemini_cloud_api
from API__Single_Generation import chatgpt_three_platforms
from API__Single_Generation import codelama_server_standard_load
from API__Single_Generation import codelama_server_standard_inference


# Load environment variables from the .env file.
# This is usually used to store sensitive information such as API keys safely.
load_dotenv()

# Initialize the OpenAI client.
# It automatically uses the OPENAI_API_KEY value from environment variables.
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure the logging system.
# Error messages are written to the file named context_error_log.log.
# The logging level is set to ERROR.
# When processing a large amount of data, the program records API call failures
# instead of crashing immediately.
logging.basicConfig(filename="context_error_log.log", level=logging.ERROR)

# Define a list to collect the IDs of test cases that encounter errors during processing.
error_ids = []


# Define all required input and output file paths.
# Before actual execution, replace these placeholder paths with your local file paths.
test_cfg_embeddings_file_path = r"PIE_data\PIE_data_embeddings.csv"
train_cfg_embeddings_file_path = r"embeddings\train_embeddings.csv"

test_dataset_file_path = r".\PIE_Cpp_009_CodeLlama13B__X.csv"
train_dataset_file_path = r"data\train_dataset.json"
train_cfg_file_path = r"data\cfg_dataset\train_cfg_dataset.json"

code_analysis_file_path = r"data\code_analysis.json"


# Define the model to be tested.
if "CodeLlama13B" in test_dataset_file_path:
    model_name = "./CodeLlama-13b-Instruct-hf"
    codelama_tokenizer, codelama_model = codelama_server_standard_load(model_name=model_name)
elif "CodeLlama34B" in test_dataset_file_path:
    model_name = "./CodeLlama-34b-Instruct-hf"
    codelama_tokenizer, codelama_model = codelama_server_standard_load(model_name=model_name)
elif "DeepSeekV32" in test_dataset_file_path:
    model_name = "deepseek-ai/DeepSeek-V3.2-Exp"
elif "Gemini" in test_dataset_file_path:
    model_name = "gemini-2.5-flash-nothinking"
elif "GPT3" in test_dataset_file_path:
    model_name = "gpt-3.5-turbo-1106"


# Main control function.
# It loads data, computes vector similarity, iterates over the test set,
# and saves the generated results.
def main(
    test_cfg_embeddings_file_path,
    train_cfg_embeddings_file_path,
    test_dataset_file_path,
    train_dataset_file_path,
    train_cfg_file_path,
    code_analysis_file_path,
    output_dataset_file_path,
):
    # 1. Load embedding data in CSV format.
    test_cfg_embeddings_df = pd.read_csv(test_cfg_embeddings_file_path)
    test_cfg_embeddings_df = test_cfg_embeddings_df[:2]
    train_cfg_embeddings_df = pd.read_csv(train_cfg_embeddings_file_path)

    # 2. Load each JSON dataset.
    # Dictionary comprehensions convert lists into dictionaries keyed by id,
    # so later lookups by ID can be completed in O(1) time.
    with open(train_dataset_file_path, "r") as file:
        train_dataset_dict = {entry["id"]: entry for entry in json.load(file)}

    with open(train_cfg_file_path, "r") as file:
        train_cfg_dict = {entry["id"]: entry for entry in json.load(file)}

    with open(code_analysis_file_path, "r") as file:
        # Note: the value of the analysis field is directly extracted as the dictionary value.
        code_analysis_dict = {entry["id"]: entry["analysis"] for entry in json.load(file)}

    # Assume the test table is saved as a CSV file.
    # If the original table is in pickle format, replace read_csv with pd.read_pickle.
    test_df = pd.read_csv(test_dataset_file_path)
    # test_df = test_df[:2]
    # Process only the first two records to speed up testing.
    # Remove this line in the official run to process all data.

    # Rename the specified columns to the field names corresponding to the original JSON data.
    test_df = test_df.rename(
        columns={
            "712_idx": "id",
            "input": "source_code",
            "target": "optimized_code",
        }
    )

    # Convert the DataFrame into a list of dictionaries.
    # Force the id key to be in string format to align with the later logic.
    test_dataset_dict = {
        str(entry["id"]): entry for entry in test_df.to_dict(orient="records")
    }

    # 3. Data preprocessing:
    # Convert vector data stored as strings back into NumPy arrays.
    # When the CSV is read, columns containing lists are treated as strings.
    # Here, json.loads parses each string into a Python list and then converts it to np.array.
    test_cfg_embeddings_df["source_cfg_embeddings"] = test_cfg_embeddings_df[
        "source_cfg_embeddings"
    ].apply(lambda x: np.array(json.loads(x)))
    train_cfg_embeddings_df["source_cfg_embeddings"] = train_cfg_embeddings_df[
        "source_cfg_embeddings"
    ].apply(lambda x: np.array(json.loads(x)))

    # Store mappings from id to predicted code for each of the five generations.
    predictions_maps = {i: {} for i in range(1, 6)}
    total_entries = len(test_cfg_embeddings_df)

    # 4. Iterate over each test entry for processing.
    for index, test_entry in test_cfg_embeddings_df.iterrows():
        test_id = test_entry["id"]

        # Retrieve the code to be optimized and the ground-truth optimized code
        # by dictionary key. The ID is converted to string to avoid type mismatches.
        slow_code = test_dataset_dict[str(test_id)]["source_code"]
        fast_code = test_dataset_dict[str(test_id)]["optimized_code"]

        # 5. Retrieve the most similar training sample based on vector similarity.
        # This is the core RAG logic.
        # Reshape the current test sample vector into (1, n_features)
        # to satisfy the input requirements of sklearn.
        test_embedding = test_entry["source_cfg_embeddings"].reshape(1, -1)

        # Stack all training vectors into a two-dimensional matrix:
        # (n_samples, n_features).
        train_embeddings = np.stack(train_cfg_embeddings_df["source_cfg_embeddings"].values)

        # Compute cosine similarity between the current test vector and all training vectors.
        similarities = cosine_similarity(test_embedding, train_embeddings)

        # Find the index of the training sample with the highest similarity score.
        most_similar_index = similarities.argmax()

        # Use the index to obtain the ID of the corresponding most similar sample.
        most_similar_id = train_cfg_embeddings_df.iloc[most_similar_index]["id"]

        # 6. Use the retrieved similar sample ID to extract the corresponding code,
        # CFG labels, and analysis text. These will be used as references for the AI.
        similar_pair_slow_code = train_dataset_dict[str(most_similar_id)]["source_code"]
        similar_pair_fast_code = train_dataset_dict[str(most_similar_id)]["optimized_code"]
        cfg_label_info = train_cfg_dict[str(most_similar_id)]["labels"]
        retrieved_code_analysis_text = code_analysis_dict[str(most_similar_id)]

        # 7 and 8. Call the optimization function five times and save each result
        # into the corresponding prediction dictionary.
        for i in range(1, 6):
            generated_code = optimize_code_with_llm(
                slow_code,
                similar_pair_slow_code,
                similar_pair_fast_code,
                cfg_label_info,
                retrieved_code_analysis_text,
            )
            if generated_code is not None:
                predictions_maps[i][test_id] = generated_code
            else:
                # If generation fails, write None to avoid misalignment.
                predictions_maps[i][test_id] = "None"

                # Record which iteration failed.
                error_ids.append(f"{test_id}_iter_{i}")
                logging.error(f"Error processing entry with ID {test_id} at iteration {i}")

        # Print the current processing progress in the console.
        print(f"Processed {index + 1}/{total_entries}")

    # 9. Map the generated results back to the original test DataFrame
    # and save the final table as a CSV file.
    # The map method automatically matches results in the dictionary by id
    # and creates new columns.
    for i in range(1, 6):
        test_df[f"AutoPatch_G5__Predict_Fast_code_{i}"] = test_df["id"].map(
            predictions_maps[i]
        )

    # Save as a CSV file.
    # index=False prevents the row index from being saved as an extra column.
    test_df.to_csv(output_dataset_file_path, index=False)

    # 10. Summary report after processing.
    if error_ids:
        print("\nThe following IDs encountered errors:")
        print(error_ids)
    else:
        print("\nAll entries were processed successfully!")


# Define the core function for calling an LLM to optimize code.
# Parameter descriptions:
# - code_to_be_optimized: the target C++ code to be optimized.
# - similar_source_code: the most similar original code retrieved from the training set.
# - similar_optimized_code: the optimized version corresponding to the similar code.
# - cfg_labels: difference labels of the Control Flow Graph.
# - code_analysis_text: deep analysis text for the similar code.
def optimize_code_with_llm(
    code_to_be_optimized,
    similar_source_code,
    similar_optimized_code,
    cfg_labels,
    code_analysis_text,
):
    # Build the prompt to be sent to the Large Language Model.
    # This prompt defines the AI role as a C++ code optimization expert,
    # provides context such as CFG differences and optimization examples,
    # and finally gives a specific instruction that the AI should return
    # only the optimized code without additional explanations.
    prompt = f"""
You are an expert in C++ code optimization, with a deep understanding of optimizing code by analyzing Control Flow Graphs (CFGs) and implementing efficient coding practices. Your task is to generate an optimized version of the provided code using the reference code and its optimized counterpart as examples, along with insights from their CFG differences and analysis.

**Context of CFG and CFG Differences:**
The CFG represents the sequence and conditions in which code blocks are executed, showing all possible paths that execution could take in the code. CFG differences indicate how the control flow has changed between the original and optimized versions, highlighting areas where execution can be streamlined or redundancies reduced. Use these differences to identify redundant operations, unnecessary branches, or bottlenecks in the provided code.

**Instructions for Using CFG Differences and Analysis to Optimize Code:**
1. Look for repetitive code segments or loops in the CFG that have been condensed or removed in the optimized reference. Apply similar techniques to minimize redundant calculations or loops in the code to optimize.
2. Focus on eliminating unnecessary branches and simplifying conditions where CFG shows reduced nodes in the optimized version.
3. Where possible, reduce the number of intermediate variables, limit memory usage, and improve inlining to match the optimized CFG structure.

Use the CFG differences, labels, and analysis provided to make structural and algorithmic improvements in the code that will yield a faster, more memory-efficient version.

**Reference Code:**
Original Code:
{similar_source_code}

Optimized Code:
{similar_optimized_code}

CFG Differences/Labels:
{cfg_labels}

Analysis:
{code_analysis_text}

**Code to Optimize:**
{code_to_be_optimized}

Return only the optimized code, with no additional comments or explanations.
    """

    try:
        model_response_text_list = generate_with_api(
            input_role_prompt=(
                "You are a highly skilled assistant providing optimized code "
                "without additional explanations."
            ),
            input_prompt_text=prompt,
            api_key_index=0,
            should_print_prompt=False,
        )

        # Store the LLM output in a temporary field for later validation.
        processed_code = postprocess_cot_generated_code_py_cpp(
            model_response_text_list[0],
            prefix="cpp",
        )

        # Extract the generated text and remove possible leading or trailing whitespace.
        return processed_code.strip()
    except Exception as e:
        # If the network request fails, the API rate limit is reached,
        # or any other exception occurs, catch the exception.
        # Record it in the log file, print it to the console, and return None
        # to indicate failure.
        logging.error(f"Error processing optimization. Exception: {e}")
        print(f"Error processing optimization. Exception: {e}")
        return None


def generate_with_api(
    input_role_prompt,
    input_prompt_text,
    api_key_index,
    should_print_prompt,
):
    if model_name in ["./CodeLlama-34b-Instruct-hf", "./CodeLlama-13b-Instruct-hf"]:
        model_response_text_list = codelama_server_standard_inference(
            model=codelama_model,
            input_role_prompt=input_role_prompt,
            input_prompt_text=input_prompt_text,
            generated_code_count=1,
            temperature=0.7,
            max_length=1024,
            should_print_prompt=should_print_prompt,
            tokenizer=codelama_tokenizer,
        )

    elif model_name == "CodeGeneration2/CodeLlama-34b-Instruct-hf":
        model_response_text_list = codelama_deepinfra(
            api_key_index=api_key_index,
            model_name=model_name,
            input_role_prompt=input_role_prompt,
            input_prompt_text=input_prompt_text,
            generated_code_count=1,
            temperature=0.7,
            should_print_prompt=should_print_prompt,
        )

    elif "gemini" in model_name:
        model_response_text_list = gemini_cloud_api(
            platform="YunwuAPI",
            api_key_index=api_key_index,
            model_name=model_name,
            input_role_prompt=input_role_prompt,
            input_prompt_text=input_prompt_text,
            generated_code_count=1,
            temperature=0.7,
            max_length=1024,
            should_print_prompt=should_print_prompt,
        )

    elif False and "gemini" in model_name:
        model_response_text_list, average_logprob_list = gemini_official(
            api_key_index=api_key_index,
            model_name=model_name,
            input_role_prompt=input_role_prompt,
            input_prompt_text=input_prompt_text,
            generated_code_count=1,
            temperature=0.7,
            max_length="NoLimit",
            thinking_budget=-404,
            return_logprob=False,
            should_print_prompt=should_print_prompt,
        )

    elif model_name in [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-4-1106-preview",
        "gpt-4o-mini",
        "gpt-4.1-nano",
    ]:
        model_response_text_list, average_logprob_list = chatgpt_three_platforms(
            platform="YunwuAPI",
            api_key_index=api_key_index,
            model_name=model_name,
            input_role_prompt=input_role_prompt,
            input_prompt_text=input_prompt_text,
            generated_code_count=1,
            temperature=0.7,
            max_length=1024,
            return_logprob=False,
            should_print_prompt=should_print_prompt,
        )

    elif model_name in ["deepseek-chat", "deepseek-reasoner", "deepseek-ai/DeepSeek-V3.2-Exp"]:
        if model_name in ["deepseek-chat", "deepseek-reasoner"]:
            platform_name = "DeepSeekOfficial"
        elif model_name in ["deepseek-ai/DeepSeek-V3.2-Exp"]:
            platform_name = "SiliconFlowAPI"

        (
            model_response_text_list,
            full_model_response_dict,
            reasoning_text,
            average_logprob,
        ) = deepseek_dual_platform(
            platform=platform_name,
            api_key_index=api_key_index,
            model_name=model_name,
            input_role_prompt=input_role_prompt,
            input_prompt_text=input_prompt_text,
            temperature=0.7,
            max_length=1024,
            should_print_prompt=should_print_prompt,
        )

    return model_response_text_list


def postprocess_cot_generated_code_py_cpp(raw_code_str, index=0, prefix="python"):
    code_str = raw_code_str.strip()

    for prefix in [
        "python",
        "Python",
        "cpp",
        "Cpp",
        "c++",
        "C++",
        " python",
        " Python",
        " cpp",
        " Cpp",
        " c++",
        " C++",
        "css",
        "bash",
        "markdown",
        "perl",
        "php",
        "scss",
        "yaml",
        "sh",
        "sql",
        "C",
        "c",
        "END_ERROR_MARKER",
    ]:
        if (f"```{prefix}" in code_str) or (f"```\n{prefix}" in code_str):
            break
        if prefix == "END_ERROR_MARKER":
            print(
                f"\n\n### Prefix error: {prefix}, index: {index}. "
                f"code_str[:20]: {code_str}"
            )

    # Handle the case with a single triple-backtick marker.
    if code_str.count("```") == 1:
        if f"```{prefix}" in code_str:
            code_str = code_str.split(f"```{prefix}")[1].strip()
        elif f"```\n{prefix}" in code_str:
            code_str = code_str.split(f"```\n{prefix}")[1].strip()
        elif "```\n" in code_str:
            code_str = code_str.split("```\n")[1].split("```")[0].strip()
            print(
                f"\n\n### Single marker: check whether the first line has an error. "
                f"index: {index}\ncode_str[:20]: {code_str[:20]}"
            )
        else:
            print(
                f"\n\n### Manual distinction is required. index: {index}\n"
                f"raw_code_str:\n{raw_code_str}"
            )
            code_str = code_str.split("```")[1].split("```")[0].strip()
        assert "```" not in code_str

    # Handle the case with two triple-backtick markers.
    elif code_str.count("```") == 2:
        if f"```{prefix}" in code_str:
            code_str = code_str.split(f"```{prefix}")[1].split("```")[0].strip()
        elif f"```\n{prefix}" in code_str:
            code_str = code_str.split(f"```\n{prefix}")[1].split("```")[0].strip()
        elif "```\n" in code_str:
            code_str = code_str.split("```\n")[1].split("```")[0].strip()
            print(
                f"\n\n### Two markers: check whether the first line has an error. "
                f"index: {index}\ncode_str[:20]: {code_str[:20]}"
            )
        else:
            print(
                f"\n\n### Manual distinction is required. index: {index}\n"
                f"raw_code_str:\n{raw_code_str}"
            )
            code_str = code_str.split("```")[1].split("```")[0].strip()
        assert "```" not in code_str

    # Handle the case with more than two triple-backtick markers.
    elif code_str.count("```") > 2:
        if f"```{prefix}" in code_str:
            code_str = code_str.split(f"```{prefix}")[1].split("```")[0].strip()
        if f"```\n{prefix}" in code_str:
            code_str = code_str.split(f"```\n{prefix}")[1].split("```")[0].strip()

        if "```" in code_str:
            code_str = code_str.split("```")[1]
            if (not code_str) or (code_str[0] != "\n"):
                print(
                    f"\n\n### Error: multiple triple-backtick markers. index: {index}\n"
                    f"raw_code_str:\n{raw_code_str}"
                )
                print(f"\n\n### Processed code_str:\n{code_str}")
            code_str = code_str.strip()

        if "```" in code_str:
            print(
                f"\n\n### Error: unresolved multiple triple-backtick markers. "
                f"index: {index}\nraw_code_str:\n{raw_code_str}"
            )
            print(f"\n\n### Processed code_str:\n{code_str}")

    if code_str == "" or code_str == "nan":
        print(
            f"\n### code_str is empty. index: {index}\n"
            f"raw_code_str:\n{raw_code_str}"
        )
        code_str = "pass"

    return code_str.strip()


# Standard Python script entry-point check.
if __name__ == "__main__":
    # Automatically generate a new output filename.
    prefix_path = "__".join(test_dataset_file_path.split("__")[:-1])
    pie_index = test_dataset_file_path.split("__")[-2].split("_")[-2]
    prefix_path = prefix_path.replace(pie_index, str(int(pie_index) + 1).zfill(3))
    output_dataset_path = f"{prefix_path}__AutoPatch_generated.csv"

    # Execute the main program.
    main(
        test_cfg_embeddings_file_path,
        train_cfg_embeddings_file_path,
        test_dataset_file_path,
        train_dataset_file_path,
        train_cfg_file_path,
        code_analysis_file_path,
        output_dataset_path,
    )
