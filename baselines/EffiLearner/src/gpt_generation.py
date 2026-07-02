import json  # Used to read and write JSON-format data
import openai  # Used to call the OpenAI API. Note: this is the legacy syntax for openai < 1.0.0.
# import json  # This is a duplicate import. It does not affect execution but can be removed.
from tqdm import tqdm  # Used to display progress bars in the terminal for easier task monitoring
import copy  # Used to deep-copy objects and prevent data overwrite issues in multithreaded environments
from concurrent.futures import ThreadPoolExecutor  # Used to create a thread pool for multithreaded concurrency
import concurrent.futures  # Imports the core concurrency module for handling asynchronous task results
from API__Single_Generation import deepseek_dual_platform_function  # Used to call the DeepSeek API


# 1. Read the base dataset
# Open the local JSON file that contains task descriptions and test cases
with open("./datasets/dataset.json", "r") as f:
    full_dataset = json.load(f)
    full_dataset = full_dataset[:1]  # For testing convenience, only process the first entry. Comment this line out to process the full dataset.


# Define the core function that requests the API and obtains code completion results
def fetch_completion(data_entry, model):
    # Extract the small test cases from the current data entry
    test_case = data_entry["small_test_cases"]

    try:
        model_response_text_list, full_model_response_dict, chain_of_thought_text, average_log_probability = deepseek_dual_platform_function(
            platform="SiliconFlowAPI",  # Options: DeepSeekOfficial, SiliconFlowAPI
            key_index=0,
            model_name="deepseek-ai/DeepSeek-V3.2-Exp",  # Options: deepseek-chat, deepseek-reasoner, deepseek-ai/DeepSeek-V3.2-Exp
            enable_reasoning=False,
            input_role_prompt="Please generate only the code.",
            input_question_text=(
                f"Please complete Python code based on the task description and test cases. "
                f"# Task description:\n{data_entry['markdown_description']}\n{test_case}\n#Solution:\n"
            ),
            all_messages=[],
            temperature=0.01,
            max_length=1024,
            output_prompt=False,
        )

        # Write the generated text returned by the model into the "completion" field of this data entry
        data_entry["completion"] = model_response_text_list[0]

    except Exception as e:
        # Catch network exceptions, timeouts, or API service errors, and print the error details
        print(repr(e))

        # Fill in an error marker to ensure the program can continue running and to facilitate later filtering of failed samples
        data_entry["completion"] = "API Error"

    return data_entry  # Return the updated data dictionary


# 2. Request the API concurrently using multiple threads
model_list = ["gpt-4"]  # Define the list of models to test

for model in model_list:
    # Instantiate a thread pool. By default, Python usually determines the maximum number of threads based on the system.
    # You can also specify it manually through max_workers.
    with ThreadPoolExecutor() as executor:

        # --- Step A: Submit tasks ---
        # Use a dictionary comprehension to submit tasks to the thread pool.
        # executor.submit() immediately returns a Future object, which represents an operation that will complete later.
        # copy.deepcopy(entry) ensures that each worker thread receives an independent data copy, preventing data races.
        # The future_to_entry dictionary establishes a mapping: Future object -> original data entry.
        future_to_entry = {
            executor.submit(fetch_completion, copy.deepcopy(entry), model): entry
            for entry in tqdm(full_dataset)
        }

        # --- Step B: Collect results ---
        # concurrent.futures.as_completed(future_to_entry) returns an iterator.
        # Whenever any thread finishes its task, the corresponding Future is yielded immediately,
        # instead of waiting rigidly in submission order.
        for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
            # Retrieve the original data entry corresponding to the completed Future object
            entry = future_to_entry[future]

            try:
                # Get the actual return value of this task, namely the dictionary returned by fetch_completion with the completion field
                updated_entry = future.result()

                # Find the index of the original dictionary in the original dataset list
                idx = full_dataset.index(entry)

                # Replace the original data entry in the dataset list with the new dictionary containing the model-generated result
                full_dataset[idx] = updated_entry

            except Exception as e:
                # Catch exceptions that may be raised while retrieving the result
                print(repr(e))

    # 3. Save the results
    # After all data has been processed, write the complete dataset containing API results into a new local JSON file
    with open(f"./EffiBench_{model}.json", "w") as f:
        # indent=4 gives the generated JSON file hierarchical indentation for easier human reading
        json.dump(full_dataset, f, indent=4)