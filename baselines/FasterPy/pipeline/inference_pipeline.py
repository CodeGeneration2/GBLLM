import sys
from pathlib import Path

# Get the absolute path of the current file and move up two levels to find the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from typing import List
from py_formatter import Formatter
from knowledge.knowledge_base import KnowledgeBase
from prompt_generator import gen_prompt
# from patch_synthesizer import syn
from fasterpy_generator import FasterPyGenerator
import re
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound


from API__Single_Generation import DeepSeek_dual_platform_function
from API__Single_Generation import CodeLlama_deepinfra_function
from API__Single_Generation import Gemini_official_function
from API__Single_Generation import Gemini_yunwu_api_function
from API__Single_Generation import ChatGPT_three_platform_function
from API__Single_Generation import CodeLlama_server_model_standard_load_function
from API__Single_Generation import CodeLlama_server_standard_inference_function

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use only GPU 1.


# Define exit status codes.
NORMAL_EXIT = 0
EXAMPLE_NOT_FOUND = 1


# #####################################################################################################################
DEBUG = False
dataset_path = r"./DB_Py_014_CodeLlama34B__X.csv"
output_dataset_path = r""
parallel_worker_count = 1


# #####################################################################################################################
# Define the list of models to be tested.
if "CodeLlama13B" in dataset_path:
    model_name = "./CodeLlama-13b-Instruct-hf"
    codellama_tokenizer, codellama_model = CodeLlama_server_model_standard_load_function(model_name=model_name)
elif "CodeLlama34B" in dataset_path:
    model_name = "./CodeLlama-34b-Instruct-hf"
    codellama_tokenizer, codellama_model = CodeLlama_server_model_standard_load_function(model_name=model_name)
elif "DeepSeekV32" in dataset_path:
    model_name = "deepseek-ai/DeepSeek-V3.2-Exp"
elif "Gemini" in dataset_path:
    model_name = "gemini-2.5-flash-nothinking"
elif "GPT3" in dataset_path:
    model_name = "gpt-3.5-turbo-1106"


slow_code_example = """
import numpy as np
def resolve():
    MOD = 10**9 + 7
    n, m = list(map(int, input().split()))
    a = [int(eval(input())) for _ in range(m)]
    dp = np.array([1] * (n + 1))
    dp[a] = 0
    for i in range(2, n + 1):
        if dp[i] != 0:
            dp[i] = np.sum(dp[i - 2 : i]) % MOD
    print((dp[n]))
resolve()
""".strip()


# # ####################################################################################################################
# def main():
#     # Instantiate the inference pipeline.
#     # If the model path is incorrect, initialization may fail here.
#     pipeline = InferencePipeline()
#
#     # Example call.
#     # Calling this without origin_code would cause an error because origin_code is required.
#     # This is only used as a demonstration entry point.
#     fast_code = pipeline.inference(origin_code=slow_code_example)
#
#     print("Fast code:\n", fast_code)


# #####################################################################################################################
def main():
    # 1. Instantiate inference pipelines.
    pipeline_instances = []
    for _ in range(parallel_worker_count):
        pipeline_instances.append(InferencePipeline())

    # 2. Read the CSV file.
    print(f"Loading dataset: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
        # df = df[:parallel_worker_count]
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # Check whether the input column exists.
    if "input" not in df.columns:
        print("Error: the 'input' column was not found in the CSV file.")
        return

    # Initialize the five new column names to be saved.
    new_columns = [f"FasterPy_G5__Predicted_Fast_Code_{i}" for i in range(1, 6)]
    for col in new_columns:
        df[col] = ""

    # Define the processing task for a single record.
    def process_row(row_index, slow_code):
        results = []
        for _ in range(5):
            try:
                # Call the inference pipeline to generate fast code.
                fast_code = pipeline_instances[row_index % parallel_worker_count].inference(
                    origin_code=slow_code,
                    api_key_index=row_index % parallel_worker_count,
                )
                results.append(fast_code if fast_code else "")
            except Exception as e:
                print(f"Inference error at row {row_index}: {e}")
                results.append("")
        return row_index, results

    # 3. Use a thread pool for parallel processing.
    # The parallel_worker_count variable is inherited from the global variable defined above.
    print(f"Start parallel code optimization with {parallel_worker_count} threads.")
    with ThreadPoolExecutor(max_workers=parallel_worker_count) as executor:
        # Submit all row-level tasks to the thread pool.
        futures = {
            executor.submit(process_row, index, row["input"]): index
            for index, row in df.iterrows()
        }

        # Use tqdm to display a progress bar.
        for future in tqdm(as_completed(futures), total=len(futures), desc="Code optimization progress"):
            row_index, generated_codes = future.result()

            # Write the five generated results back to the corresponding DataFrame columns.
            for i, code in enumerate(generated_codes):
                df.at[row_index, new_columns[i]] = code

    # 4. Save the file to CSV.
    # Automatically generate a new file name.
    if output_dataset_path == "":
        prefix_path = "__".join(dataset_path.split("__")[:-1])
        pie_index = dataset_path.split("__")[-2].split("_")[-2]
        prefix_path = prefix_path.replace(pie_index, str(int(pie_index) + 1).zfill(3))
        final_output_dataset_path = f"{prefix_path}__FasterPy_generated.csv"
    else:
        final_output_dataset_path = output_dataset_path

    try:
        df.to_csv(final_output_dataset_path, index=False)
        print(f"\nOptimization completed. Results were successfully saved to: {final_output_dataset_path}")
    except Exception as e:
        print(f"\nFailed to save CSV: {e}")


# #####################################################################################################################
class InferencePipeline:
    """
    Code inference and optimization pipeline.

    This class is mainly responsible for receiving the original code to be optimized,
    cleaning its format, retrieving similar optimization cases from the knowledge base,
    constructing a prompt for the LLM, generating optimized code, and extracting the final code.
    """

    def __init__(self, model_path: str = "../model/deepseek-coder-6.7b-Instruct-FT"):
        # Initialize the code formatting tool used to extract the target function and its callees.
        self.formatter = Formatter()

        # Initialize the knowledge base connection.
        # "CKB" represents the specified database name, possibly Code Knowledge Base.
        self.kb = KnowledgeBase("CKB")

        # Initialize the code generator model loading and inference interface.
        # self.fpg = FasterPyGenerator(model_path)

    def is_probably_python(self, text: str) -> bool:
        """
        Use Pygments to guess whether the given text is Python code.
        """
        try:
            lexer = guess_lexer(text)
            return "Python" in lexer.name
        except ClassNotFound:
            return False

    def extract_code(self, text: str):
        """
        Extract the actual code from an LLM response.

        The code is usually wrapped in Markdown backticks.
        Note: the previous type hint may have been inaccurate because this logic treats text as a string.
        """
        # First-level matching: try to match the standard format ```python ... ```.
        pattern = r"```python\s*(.*?)```"
        code_blocks = re.findall(pattern, text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()

        # Second-level matching: if the Python language tag is missing, match code blocks without a language tag.
        pattern = r"```\s*(.*?)```"
        code_blocks = re.findall(pattern, text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()

        # Third-level matching: if the regular expressions above fail, manually split the text by ```.
        text = text.strip()
        possible_codes = text.split("```")

        # Check whether the first or last split block is likely to be Python code.
        if self.is_probably_python(possible_codes[0]):
            return possible_codes[0].strip()
        elif self.is_probably_python(possible_codes[-1]):
            return possible_codes[-1].strip()

        # If all extraction methods fail, return the entire text unchanged.
        return text

    # r_thre denotes the rate threshold, and s_thre denotes the similarity threshold.
    def inference(
        self,
        origin_code: str,
        api_key_index: int,
        target_function_name: str = "Solution",
        example_num: int = 1,
        r_thre: float = 0.5,
        s_thre: float = 0.1,
    ) -> str:
        """
        Inference workflow for optimizing a single code sample.
        """
        # 1. Input the code to be improved and specify the core function name to be improved.
        # 2. Use formatter to remove irrelevant code, keep only the specified function and its callees,
        #    and clean the code format.
        # formatted_code = self.formatter.format(origin_code, target_function_name)
        formatted_code = self.formatter.format(origin_code)
        if not formatted_code:
            print("Error when formatting code.")
            formatted_code = origin_code
            # return 0

        # 3. Perform vector retrieval from the knowledge base to find similar improved code and suggestions.
        result = self.kb.search(
            [formatted_code],
            top_k=example_num,
            similarity_thredhold=s_thre,
            rate_thredhold=r_thre,
        )
        hits = result[0]

        if len(hits) < 1:
            print("Could not find a similar improvement case.")
            improve_examples = []
            # return EXAMPLE_NOT_FOUND
        else:
            # Extract the summary and distance or similarity score of each similar case.
            improve_examples = [(hit["entity"]["summary"], hit["distance"]) for hit in hits]

        # print(improve_examples)

        # 4. Generate the prompt submitted to the LLM.
        function_signature = f"def {target_function_name}("
        # prompt = gen_prompt(origin_code, function_signature, improve_examples)
        prompt = gen_prompt(origin_code, "", improve_examples)

        if DEBUG:
            print("Prompt:\n", prompt)

        # 5. Pass the prompt to fasterpy_generator.py and call the LLM to generate optimized code.
        # ans = self.fpg([prompt])
        model_response_text = generate_data_with_api(
            system_prompt="Please generate only the code.",
            user_prompt_text=prompt,
            api_key_index=api_key_index,
            temperature=0.7,
            max_length=1024,
            should_print_prompt=DEBUG,
        )

        # 6. Filter and extract a pure code snippet from the complete model response.
        optimized_code = self.extract_code(model_response_text)

        # 7. Print the optimized code in the console.
        if DEBUG:
            print(f"=== Optimized code ===\n{optimized_code}")

        # 8. End normally and return the optimized code.
        return optimized_code

    def inference_batch(
        self,
        origin_codes: List[str],
        target_function_names: List[str] = ["Solution"],
        example_num: int = 1,
        r_thre: float = 0.5,
        s_thre: float = 0.1,
    ) -> int:
        """
        Batch inference workflow for code optimization.
        """
        # 1. Input a batch of code samples to be improved and the corresponding function names.
        # 2. Use formatter to clean and format each code sample.
        formatted_codes = []

        # Note: the original default typing hint may be problematic.
        # target_function_names should be passed as a list.
        for origin_code, target_function_name in zip(origin_codes, target_function_names):
            formatted_code = self.formatter.format(origin_code, target_function_name)
            if not formatted_code:
                print("Error when formatting code.")
                formatted_code = origin_code
            formatted_codes.append(formatted_code)

        # 3. Retrieve similar improved code and improvement cases from the knowledge base in batch.
        result = self.kb.search(
            formatted_codes,
            top_k=example_num,
            similarity_thredhold=s_thre,
            rate_thredhold=r_thre,
        )
        improve_examples_list = []

        for hits in result:
            if len(hits) < 1:
                print("Could not find a similar improvement case.")
                improve_examples_list.append([])
            else:
                improve_examples = [(hit["entity"]["summary"], hit["distance"]) for hit in hits]
                improve_examples_list.append(improve_examples)

        # 4. Generate prompts in batch.
        prompts = [
            gen_prompt(origin_code, target_function_name, improve_examples)
            for origin_code, target_function_name, improve_examples
            in zip(origin_codes, target_function_names, improve_examples_list)
        ]

        optimized_codes = []
        for prompt in prompts:
            # 5. Pass prompts one by one to fasterpy_generator.py to generate optimized code.
            #    The current design does not use underlying batch concurrency, but calls it in an outer loop.
            ans = self.fpg([prompt])

            # 6. Extract pure code.
            code = self.extract_code(ans)
            optimized_codes.append(code)

        # 7. Print all newly optimized code samples.
        for i, code in enumerate(optimized_codes):
            print(f"=== Optimized code {i} ===\n{code}")

        # 8. End normally and return 0.
        return 0


# #####################################################################################################################
def generate_data_with_api(
    system_prompt,
    user_prompt_text,
    api_key_index,
    temperature,
    max_length=1024,
    should_print_prompt=False,
):
    if model_name in ["./CodeLlama-34b-Instruct-hf", "./CodeLlama-13b-Instruct-hf"]:
        model_response_text_list = CodeLlama_server_standard_inference_function(
            model=codellama_model,
            system_prompt=system_prompt,
            input_question_text=user_prompt_text,
            generated_code_count=1,
            temperature=temperature,
            max_length=max_length,
            should_print_prompt=should_print_prompt,
            tokenizer=codellama_tokenizer,
        )

    elif model_name == "CodeGeneration2/CodeLlama-34b-Instruct-hf":
        model_response_text_list = CodeLlama_deepinfra_function(
            api_key_index=api_key_index,
            model_name=model_name,
            system_prompt=system_prompt,
            input_question_text=user_prompt_text,
            generated_code_count=1,
            temperature=temperature,
            max_length=max_length,
            should_print_prompt=should_print_prompt,
        )

    elif "gemini" in model_name:
        model_response_text_list = Gemini_yunwu_api_function(
            platform="Yunwu API",
            api_key_index=api_key_index,
            model_name=model_name,
            system_prompt=system_prompt,
            input_question_text=user_prompt_text,
            generated_code_count=1,
            temperature=temperature,
            max_length=max_length,
            should_print_prompt=should_print_prompt,
        )

    elif False and "gemini" in model_name:
        model_response_text_list, average_log_probability_list = Gemini_official_function(
            api_key_index=api_key_index,
            model_name=model_name,
            system_prompt=system_prompt,
            input_question_text=user_prompt_text,
            generated_code_count=1,
            temperature=temperature,
            max_length="No limit",
            thinking_budget=-404,
            return_log_probability=False,
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
        model_response_text_list, average_log_probability_list = ChatGPT_three_platform_function(
            platform="Yunwu API",
            api_key_index=api_key_index,
            model_name=model_name,
            system_prompt=system_prompt,
            input_question_text=user_prompt_text,
            generated_code_count=1,
            temperature=temperature,
            max_length=max_length,
            return_log_probability=False,
            should_print_prompt=should_print_prompt,
        )

    elif model_name in ["deepseek-chat", "deepseek-reasoner", "deepseek-ai/DeepSeek-V3.2-Exp"]:
        if model_name in ["deepseek-chat", "deepseek-reasoner"]:
            platform_name = "Official DeepSeek"
        elif model_name in ["deepseek-ai/DeepSeek-V3.2-Exp"]:
            platform_name = "SiliconFlow API"

        model_response_text_list, full_model_response_dict, reasoning_text, average_log_probability = (
            DeepSeek_dual_platform_function(
                platform=platform_name,
                api_key_index=api_key_index,
                model_name=model_name,
                system_prompt=system_prompt,
                input_question_text=user_prompt_text,
                temperature=temperature,
                max_length=max_length,
                should_print_prompt=should_print_prompt,
            )
        )

    return model_response_text_list[0]


# #####################################################################################################################
if __name__ == "__main__":
    main()