Prompt_Dict = {


"Prompt_Description": "Code + IO --> (Gen) NL",


# ===============================================================================================================================================================================================
"Role_Play_Python": 
            "You are an experienced software engineer and technical documentation expert, skilled in automatically generating corresponding code function descriptions based on Python source code and example input/output unit test cases."
            "Place the newly generated code function description into a block formatted as follows:\n```Description\n<code function description>\n```",
"Role_Play_Cpp": 
            "You are an experienced software engineer and technical documentation expert, skilled in automatically generating corresponding code function descriptions based on C++ source code and example input/output unit test cases."
            "Place the newly generated code function description into a block formatted as follows:\n```Description\n<code function description>\n```",


# ===============================================================================================================================================================================================
"Operation_Command_Python": 
            "Below is the Python source code and example input/output unit test cases. Based on this, please generate a concise natural language statement that summarizes the code's function description.\n\n"
            "### Code:\n```python\n{Slow_program}\n```\n\n"
            "### Related Input/Output unit test cases:\n"
            "### Please follow the above instructions and output format specifications to generate the code's function description step by step:\n",
"Operation_Command_Cpp": 
            "Below is the C++ source code and example input/output unit test cases. Based on this, please generate a concise natural language statement that summarizes the code's function description.\n\n"
            "### Code:\n```cpp\n{Slow_program}\n```\n\n"
            "### Related Input/Output unit test cases:\n"
            "### Please follow the above instructions and output format specifications to generate the code's function description step by step:\n",

# ===============================================================================================================================================================================================
# IO_Test_Cot
"IO_Format": "## Test Case {IO_ID}:\nInput: {IO_Input}\nOutput: {IO_Output}\n\n", 

}