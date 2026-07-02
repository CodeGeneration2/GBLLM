# The main entry file of the program:
# Provides an interactive command-line interface (CLI) for starting code optimization tasks.
import argparse

# Import the core inference pipeline module.
from inference_pipeline import InferencePipeline


def optimize_single_function(pipeline):
    """
    Logic branch for optimizing a single function.

    This function receives the user-provided file path and target function name,
    reads the source code, and passes it to the inference pipeline for processing.
    """
    # Get the original code file path from user input and strip surrounding spaces to avoid path errors.
    code_path = input("Please enter the path to the original code file:").strip()

    # Get the name of the specific function to optimize.
    function_name = input("Please enter the function name to optimize:").strip()

    # Safely read the specified code file using UTF-8 encoding.
    with open(code_path, "r", encoding="utf-8") as file:
        origin_code = file.read()

    # Call the inference pipeline to optimize the specified single function.
    status_code = pipeline.inference(
        origin_code=origin_code,          # Original complete code text
        target_function_name=function_name,  # Target function name to optimize
        example_num=3,                    # Number of reference cases retrieved from the knowledge base; Top-K is 3
        r_thre=0.5,                       # Rate threshold used only for knowledge base filtering
        s_thre=0.1                        # Similarity threshold used to search for similar cases
    )

    # Determine whether optimization succeeded based on the agreed return status code.
    # A status code of 0 usually indicates normal termination.
    if status_code == 0:
        print("Optimization completed!")
    else:
        print("Optimization failed.")


def optimize_multiple_functions(pipeline):
    """
    Logic branch for batch optimization of multiple functions.

    This function allows the user to enter multiple comma-separated function names
    and optimizes these functions in the same file one by one.
    """
    # Get the file path.
    code_path = input("Please enter the path to the original code file:").strip()

    # Get the list of function names to optimize.
    function_names_text = input(
        "Please enter the list of function names to optimize (comma-separated):"
    ).strip()

    # Split the string by commas and remove surrounding spaces from each function name.
    function_names = [name.strip() for name in function_names_text.split(",")]

    # Read the code file content.
    with open(code_path, "r", encoding="utf-8") as file:
        origin_code = file.read()

    # Since all target functions are in the same original file, prepare a code list
    # with the same length as function_names for the batch interface.
    origin_codes = [origin_code] * len(function_names)

    # Call the batch optimization interface of the inference pipeline.
    status_code = pipeline.inference_batch(
        origin_codes=origin_codes,              # List of original code contents
        target_function_names=function_names,   # Corresponding list of target function names
        example_num=3,
        r_thre=0.5,
        s_thre=0.1
    )

    if status_code == 0:
        print("Batch optimization completed!")
    else:
        print("Optimization failed.")


def main(model_path="../models/Qwen2.5-7B-Instruct-ft"):
    """
    Main control function.

    This function initializes the LLM inference pipeline and runs the interactive menu loop.
    By default, it loads the fine-tuned Qwen2.5-7B-Instruct model.
    """
    # Instantiate the inference pipeline and load the large model from the specified path.
    # This step may take some time and consume GPU memory.
    pipeline = InferencePipeline(model_path)

    # Start an infinite loop as the interactive CLI menu.
    while True:
        print("\n=== Code Optimization Menu ===")
        print("1. Optimize a single function")
        print("2. Optimize multiple functions")
        print("3. Exit")
        choice = input("Please select an option:").strip()

        if choice == "1":
            try:
                optimize_single_function(pipeline)
            except Exception as error:
                # Catch and print any exception that may occur when running option 1,
                # such as a missing file, to prevent the program from crashing directly.
                print(f"An error occurred in option 1: {error}")

        elif choice == "2":
            try:
                optimize_multiple_functions(pipeline)
            except Exception as error:
                # Use the same exception handling mechanism as above.
                print(f"An error occurred in option 2: {error}")

        elif choice == "3":
            print("Program exited.")
            break

        else:
            # Handle invalid menu input.
            print("Invalid option, please try again.")


if __name__ == "__main__":
    # Set up the command-line argument parser so that users can override default settings from the terminal.
    parser = argparse.ArgumentParser(description="FasterPy")

    # Add the -modelpath argument.
    # If it is not provided, the default path ../models/Qwen2.5-7B-Instruct-ft is used.
    parser.add_argument(
        "-modelpath",
        type=str,
        default="../models/Qwen2.5-7B-Instruct-ft",
        help="Local path or download URL of the model"
    )
    args = parser.parse_args()

    # Start the main function with the parsed argument.
    main(args.modelpath)

    # Logic bug note:
    # The original author called main() again without arguments here.
    # This means that when the user enters '3' to exit from the first menu,
    # the program does not actually terminate.
    # Instead, it restarts the menu using the default model path.
    # It is recommended to remove the following line.
    # main()