from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    GPTNeoForCausalLM,
    AutoModelForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
)
import torch
import json
from tqdm import tqdm
import os
import argparse
from datasets import load_dataset


def construct_prompt_template(inputs, checkpoint, model, tokenizer):
    """
    Convert text inputs into model tensors, call the model to generate responses,
    and decode and post-process the outputs.
    """
    # Although the device variable is defined here, it is not used later.
    # The code directly uses model.device instead.
    device = "cuda"

    # Set pad_token to eos_token. This is a common practice for many decoder-only
    # autoregressive models, such as GPT-style models, to avoid padding errors.
    tokenizer.pad_token = tokenizer.eos_token

    # Convert batched text inputs into the token tensor format required by the model.
    input_tokens = tokenizer.batch_encode_plus(
        inputs,
        padding=True,        # Align sequence lengths within the batch.
        return_tensors="pt", # Return PyTorch tensors.
    ).to(model.device)       # Move tensors to the device where the model is located, such as a GPU.

    # Ensure that all tensors in the input_tokens dictionary are correctly assigned
    # to the device where the model is located.
    for token_key in input_tokens:
        if torch.is_tensor(input_tokens[token_key]):
            input_tokens[token_key] = input_tokens[token_key].to(model.device)

    # Commented-out code: some models do not require token_type_ids,
    # and they can be removed using pop.
    # input_tokens.pop("token_type_ids")

    try:
        # Call the model to generate code content.
        sequences = model.generate(
            **input_tokens,
            max_new_tokens=512, # Set the maximum number of newly generated tokens to 512.
            do_sample=True      # Enable sampling mode instead of greedy search to increase output diversity.
        )

        # Decode the generated token sequences back into human-readable strings,
        # while skipping special control tokens such as </s>.
        generated_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)

        # Iterate over the generated texts and remove the input prompt,
        # keeping only the newly generated content.
        for i in range(len(generated_texts)):
            if inputs[i] in generated_texts[i]:
                generated_texts[i] = generated_texts[i].replace(inputs[i], "")

    except Exception:
        # Exception handling: if OOM or other errors occur during generation,
        # return a list of empty strings as a fallback.
        generated_texts = ["" for _ in range(len(inputs))]

    return generated_texts


def fetch_completion(data_entry_list, model, checkpoint, tokenizer):
    """
    Build prompts according to different dataset formats, including EffiBench,
    HumanEval, and MBPP, and call the generation function to obtain results.
    """
    input_batches = []

    # Iterate over each data entry in the current batch and assemble different
    # prompt instructions according to the dataset type.
    for data_entry in data_entry_list:
        if data_entry["dataset"] == "EffiBench":
            test_case = data_entry["small_test_cases"]

            # Concatenate the task description and test cases, and ask the model
            # to output the solution.
            input_batches.append(
                f"Please complete Python code based on the task description and test cases. "
                f"# Task description:\n{data_entry['markdown_description']}\n{test_case}\n#Solution:\n"
            )

        elif data_entry["dataset"] == "HumanEval":
            # The HumanEval prompt is already included in the dataset, so it is
            # directly concatenated here.
            input_batches.append(
                f"Please complete Python code based on the task description. "
                f"# Task description:\n{data_entry['prompt']}\n#Solution:\n"
            )

        elif data_entry["dataset"] == "MBPP":
            tests = "\n".join(data_entry["test_list"])

            # MBPP requires concatenating the prompt and multiple test cases.
            input_batches.append(
                f"Please complete Python code based on the task description and test cases. "
                f"# Task description:\n{data_entry['prompt']}\n{tests}\n#Solution:\n"
            )

    # Call the core generation function to obtain batched generated code.
    completion_list = construct_prompt_template(input_batches, checkpoint, model, tokenizer)

    # Write the generated code back into the "completion" field of the original data dictionaries.
    for i in range(len(data_entry_list)):
        data_entry_list[i]["completion"] = completion_list[i]

    return data_entry_list


if __name__ == "__main__":
    # Configure command-line argument parsing.
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--checkpoint",
        type=str,
        default="m-a-p/OpenCodeInterpreter-DS-33B",
        required=True,
    ) # Model weight path or HuggingFace ID.
    argument_parser.add_argument("--batch_size", type=int, default=16) # Batch size.
    argument_parser.add_argument("--dataset", type=str, required=True) # Specify the dataset name to test.
    args = argument_parser.parse_args()

    checkpoint = args.checkpoint
    batch_size = args.batch_size
    print("Checkpoint: ", checkpoint)

    # Extract the last part of the model name for naming saved files later.
    if "/" in checkpoint:
        end_name = checkpoint.split("/")[-1]

    # Load the corresponding dataset according to the input argument.
    if args.dataset == "EffiBench":
        # Read EffiBench from a local JSON file.
        with open("./datasets/dataset.json", "r") as file:
            dataset = json.load(file)

    elif args.dataset == "HumanEval":
        # Use the EvalPlus enhanced HumanEval dataset, loaded directly from
        # the HuggingFace Datasets library.
        dataset = load_dataset("evalplus/humanevalplus", split="test")

    elif args.dataset == "MBPP":
        # Use the EvalPlus enhanced MBPP dataset.
        dataset = load_dataset("evalplus/mbppplus", split="test")

    # Add a dataset-name label to each data entry in the dataset.
    for i in range(len(dataset)):
        dataset[i]["dataset"] = args.dataset

    # Load the model:
    # device_map="auto": automatically distribute model layers across available GPU memory.
    # trust_remote_code=True: allow execution of custom Python code from the model repository.
    # torch_dtype=torch.float16: load the model in half precision to greatly reduce memory usage and accelerate inference.
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Load the corresponding tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    # Use tqdm to display a progress bar and iterate over the entire dataset
    # batch by batch for inference.
    for i in tqdm(range(0, len(dataset), batch_size)):
        # Send the current batch into fetch_completion and update the generated
        # completion results back into the dataset.
        dataset[i:i + batch_size] = fetch_completion(
            dataset[i:i + batch_size],
            model,
            checkpoint,
            tokenizer,
        )

    # Extract the short model name again. This is redundant because it was already extracted earlier.
    end_name = checkpoint.split("/")[-1]

    # Save the complete dataset containing model-generated results into a local
    # JSON file for subsequent correctness evaluation.
    with open(f"../results/{args.dataset}_{end_name}.json", "w") as file:
        json.dump(dataset, file, indent=4)