import os
import argparse
from tqdm import tqdm
import json
import pprint
import pandas as pd
import time
from datetime import datetime, timedelta

from API__LLMs import call_deepseek
from API__LLMs import call_openai_chatgpt
from API__LLMs import call_gemini
from API__LLMs import call_code_llama

# Global default core index
GLOBAL_CORE_INDEX = 20

# API key lists
deepseek_api_keys = []
chatgpt_api_keys = []
gemini_api_keys = []
codellama_api_keys = []

# Role prompt options
role_prompt_dict = {
    'english_generate_nl': (
        "You are a software developer who needs to extract and output the code's functionality description. "
        "Only reply with the functionality description related to the code. "
        "Do not explain anything or include any additional notes—simply print the code functionality description. "
        "Place the code functionality description in a code block using the following format:\n```\nCode functionality description\n```"
    ),
    'english_triple_quotes': (
        "You are a software developer and now you will help to improve code efficiency. "
        "Only reply with the source code. "
        "Do not explain anything and include any extra instructions, only print the source code. "
        "Make the code in code blocks:\n```python\n```"
    ),
}

# Prompt template options
prompt_template_dict = {
    'english_generate_nl_io': (
        "### Below is a piece of code along with its corresponding example input/output unit tests. "
        "Please extract and output the code’s functionality description based on this information.\n\n"
        "### Code:\n```\n{Slow_program}\n```\n\n"
        "### Code Functionality Description:\n"
    ),
    'english_triple_quotes_cfg_io_use_nl': (
        "### Below is a slow program. Please optimize it and provide a faster version. "
        "For reference, a Control Flow Graph (CFG) framework for the faster version, "
        "a description of its functionality, and example input/output unit tests are provided.\n\n"
        "### Slow Program:\n```\n{Slow_program}\n```\n\n"
        "### Control Flow Graph Framework of the Faster Version:\n```\n{Refer_Fast_CFG}\n```\n\n"
        "### Code Functionality Description:\n```\n{Code_Function_Description}\n```\n\n"
        "### Optimized Version:\n"
    ),
    'io_test_template': (
        "### Example {IO_index}:\nInput: {IO_input}\nOutput: {IO_output}\n\n"
    ),

    # Prompt template options for ablation experiments
    # Ablation: triple-quoted block with CFG and IO tests
    'ablation_quotes_cfg_io': (
        "### Below is a slow program. Please optimize it and provide a faster version. "
        "A control flow graph (CFG) framework for a more efficient version is provided for reference.\n\n"
        "### Slow Program:\n```\n{Slow_program}\n```\n\n"
        "### Control Flow Graph Framework of the Faster Version:\n```\n{Refer_Fast_CFG}\n```\n\n"
        "### Optimized Version:\n"
    ),

    # Ablation: no CFG, triple-quoted block with IO tests and functionality description
    'ablation_no_cfg_quotes_io_with_nl': (
        "### Below is a slow program. Please optimize it and provide a faster version. "
        "For reference, a description of its functionality and example input/output unit tests are provided.\n\n"
        "### Slow Program:\n```\n{Slow_program}\n```\n\n"
        "### Code Functionality Description:\n```\n{Code_Function_Description}\n```\n\n"
        "### Optimized Version:\n"
    ),

    # Ablation: include IO tests but no CFG, generate only natural language description
    'ablation_io_no_outputs_generate_nl': (
        "### Below is a piece of code along with its corresponding example input/output unit tests. "
        "Please extract and output the code’s functionality description based on this information.\n\n"
        "### Code:\n```\n{Slow_program}\n```\n\n"
        "### Code Functionality Description:\n"
    ),

    # Ablation: include IO tests and CFG, triple-quoted block with functionality description
    'ablation_io_no_outputs_quotes_cfg_use_nl': (
        "### Below is a slow program. Please optimize it and provide a faster version. "
        "For reference, a Control Flow Graph (CFG) framework for the faster version, a description "
        "of its functionality, and example input/output unit tests are provided.\n\n"
        "### Slow Program:\n```\n{Slow_program}\n```\n\n"
        "### Control Flow Graph Framework of the Faster Version:\n```\n{Refer_Fast_CFG}\n```\n\n"
        "### Code Functionality Description:\n```\n{Code_Function_Description}\n```\n\n"
        "### Optimized Version:\n"
    ),
}


def get_hyperparameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--base_data_path", default=r'', type=str)
    parser.add_argument("--output_path", default=r'', type=str)
    parser.add_argument("--role_key", default="english_triple_quotes", type=str,
                        help='[english_generate_nl, english_triple_quotes]')
    parser.add_argument("--prompt_key", default="english_triple_quotes_cfg_io_use_nl", type=str,
                        help='[english_generate_nl_io, english_triple_quotes_cfg_io_use_nl]')
    parser.add_argument("--io_key", default="io_test_template", type=str)
    parser.add_argument("--code_column_name", default="Slow_program", type=str)
    parser.add_argument("--cfg_column_name", default="Refer_Fast_CFG", type=str)
    parser.add_argument("--nl_column_name", default="Code_Function_Description_T0_01", type=str)
    parser.add_argument("--num_responses", default=5, type=int)
    args = parser.parse_args()
    return args

def set_additional_parameters(args, core_index):
    if core_index == -1:
        index = GLOBAL_CORE_INDEX
    else:
        index = core_index

    

    # Adjust output path and log settings based on model
    if args.model_name == "gpt-3.5-turbo":
        args.output_path = f"{args.output_path}_GPT35"
        args.return_logprobs = True
    elif args.model_name == "gpt-4-1106-preview":
        args.output_path = f"{args.output_path}_GPT4"
        args.return_logprobs = True
    elif args.model_name == "gemini-1.5-pro":
        args.output_path = f"{args.output_path}_Gemini"
    elif args.model_name == "CodeGeneration2/CodeLlama-34b-Instruct-hf":
        args.output_path = f"{args.output_path}_CodeLlama"

    # Assign the chosen prompts and paths
    args.input_role = role_prompt_dict[args.role_key]
    args.input_prompt_template = prompt_template_dict[args.prompt_key]
    args.io_prompt_template = prompt_template_dict[args.io_key]
    args.log_path = f"{args.output_path}_{args.prompt_key}"
    args.output_path = f"{args.output_path}_{args.prompt_key}/test"

    # Set default temperature based on number of responses
    if args.num_responses == 1:
        args.temperature = 0.01
    elif args.num_responses >= 5:
        args.temperature = 0.7

    args.core_index = index
    return args

def main(core_index=-1):
    args = get_hyperparameters()
    args = set_additional_parameters(args, core_index)
    args_dict = vars(args)
    print(pprint.pformat(args_dict))

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    # Save args dictionary to log
    with open(f"{args.log_path}/args.txt", "w", encoding="utf-8") as log_file:
        json.dump(args_dict, log_file, ensure_ascii=False, indent=4)

    # Load baseline data
    if args.model_name in ["gpt-3.5-turbo", "gemini-1.5-pro", "CodeGeneration2/CodeLlama-34b-Instruct-hf"]:
        df = pd.read_csv(args.base_data_path, encoding='ISO-8859-1')
    else:
        df = pd.read_csv(args.base_data_path)

    # Prepare code lists
    inefficient_code_list = df[args.code_column_name].tolist()
    if 'cfg' in args.prompt_key:
        reference_fast_cfg = df[args.cfg_column_name].tolist()
    if 'io' in args.prompt_key:
        io_unit_test_dicts = df['Public_IO_unit_tests'].tolist()
    if 'nl' in args.prompt_key:
        code_function_nl_descriptions = df[args.nl_column_name].tolist()

    # Adjust end index if 'end'
    if args.end_index == 'end':
        args.end_index = len(inefficient_code_list)

    # Process each code example
    for idx in tqdm(range(args.start_index, args.end_index)):
        idx_str = str(idx).zfill(4)
        # Skip if output already exists
        if os.path.exists(f"{args.output_path}/{idx_str}.txt") or os.path.exists(f"{args.output_path}/{idx_str}_0.txt"):
            continue

        # Build the prompt for this example
        if 'cfg' in args.prompt_key and 'nl' in args.prompt_key:
            prompt = args.input_prompt_template.format(
                Slow_program=inefficient_code_list[idx].strip(),
                Refer_Fast_CFG=reference_fast_cfg[idx].strip(),
                Code_Function_Description=str(code_function_nl_descriptions[idx]).strip()
            )
        elif 'cfg' in args.prompt_key:
            prompt = args.input_prompt_template.format(
                Slow_program=inefficient_code_list[idx].strip(),
                Refer_Fast_CFG=reference_fast_cfg[idx].strip()
            )
        elif 'nl' in args.prompt_key:
            prompt = args.input_prompt_template.format(
                Slow_program=inefficient_code_list[idx].strip(),
                Code_Function_Description=str(code_function_nl_descriptions[idx]).strip()
            )
        else:
            prompt = args.input_prompt_template.format(
                Slow_program=inefficient_code_list[idx].strip()
            )

        # Append IO tests if needed
        if 'io' in args.prompt_key:
            io_dict = eval(io_unit_test_dicts[idx])
            inputs = io_dict['inputs']
            outputs = io_dict['outputs']
            io_str = ""
            for i, (inp, outp) in enumerate(zip(inputs, outputs)):
                inp = inp.strip()
                outp = outp.strip()
                if not is_numeric_string(inp):
                    inp = f"'{inp}'"
                if not is_numeric_string(outp):
                    outp = f"'{outp}'"
                io_str += args.io_prompt_template.format(IO_index=i+1, IO_input=inp, IO_output=outp)
            if 'generate_nl' in args.prompt_key:
                prompt = prompt.replace(
                    "### Code Functionality Description",
                    f"{io_str}### Code Functionality Description"
                )
            else:
                prompt = prompt.replace(
                    "### Optimized Version",
                    f"{io_str}### Optimized Version"
                )

        # Call the appropriate API and save results
        if args.model_name in ["deepseek-chat", "deepseek-reasoner"]:
            output_text, chain_of_thought, probabilities = call_deepseek(
                api_key=args.api_key,
                model_name=args.model_name,
                input_role=args.input_role,
                input_prompt=prompt
            )
            with open(f"{args.output_path}/{idx_str}.txt", 'w', encoding='utf-8', newline='\n') as f:
                f.write(output_text)
            if chain_of_thought:
                with open(f"{args.output_path}/{idx_str}_thought.txt", 'w', encoding='utf-8', newline='\n') as f:
                    f.write(chain_of_thought)

        elif args.model_name in ["gpt-3.5-turbo", "gpt-4-1106-preview"]:
            responses, thoughts, nested_logprobs, avg_logprobs = call_openai_chatgpt(
                api_key=args.api_key,
                model_name=args.model_name,
                input_role=args.input_role,
                input_prompt=prompt,
                num_responses=args.num_responses,
                temperature=args.temperature,
                return_logprobs=args.return_logprobs
            )
            for i, resp in enumerate(responses):
                with open(f"{args.output_path}/{idx_str}_{i}.txt", 'w', encoding='utf-8', newline='\n') as f:
                    f.write(resp)
                if thoughts:
                    with open(f"{args.output_path}/{idx_str}_{i}_thought.txt", 'w', encoding='utf-8', newline='\n') as f:
                        f.write(thoughts[i])
                if args.return_logprobs:
                    with open(f"{args.output_path}/{idx_str}_{i}_logprobs.txt", 'w', encoding='utf-8', newline='\n') as f:
                        f.write(str(nested_logprobs[i]))
                    with open(f"{args.output_path}/{idx_str}_{i}_avg_logprobs.txt", 'w', encoding='utf-8', newline='\n') as f:
                        f.write(str(avg_logprobs[i]))

        elif args.model_name == "gemini-1.5-pro":
            responses, avg_logprobs = call_gemini(
                api_key=args.api_key,
                model_name=args.model_name,
                input_role=args.input_role,
                input_prompt=prompt,
                num_responses=args.num_responses,
                temperature=args.temperature
            )
            for i, resp in enumerate(responses):
                with open(f"{args.output_path}/{idx_str}_{i}.txt", 'w', encoding='utf-8', newline='\n') as f:
                    f.write(resp)
                with open(f"{args.output_path}/{idx_str}_{i}_avg_logprobs.txt", 'w', encoding='utf-8', newline='\n') as f:
                    f.write(str(avg_logprobs[i]))

        elif args.model_name == "CodeGeneration2/CodeLlama-34b-Instruct-hf":
            responses = call_code_llama(
                api_key=args.api_key,
                model_name=args.model_name,
                input_role=args.input_role,
                input_prompt=prompt,
                num_responses=args.num_responses,
                temperature=args.temperature
            )
            for i, resp in enumerate(responses):
                with open(f"{args.output_path}/{idx_str}_{i}.txt", 'w', encoding='utf-8', newline='\n') as f:
                    f.write(resp)


def is_numeric_string(s):
    """
    Check if the given string represents a numeric value.
    Returns True if it can be converted to float, False otherwise.
    """
    s_clean = s.strip().replace("\n", "").replace(" ", "")
    try:
        float(s_clean)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
