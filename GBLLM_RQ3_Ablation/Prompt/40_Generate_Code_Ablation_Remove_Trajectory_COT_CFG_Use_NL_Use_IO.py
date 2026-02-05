Prompt_Dict = {


"Prompt_Description": "Code + NL_COT + CFG + I/O + Slow/Mid/Fast --> Efficient Code",


# ===============================================================================================================================================================================================
"Role_Play_Python": 
            "You are a software developer, and now you will help improve code efficiency. "
            "Please generate faster code according to the instructions and output format specifications. "
            "The optimized code should be in code block format:\n```python\n<code>\n```", 
"Role_Play_Cpp": 
            "You are a software developer, and now you will help improve code efficiency. "
            "Please generate faster code according to the instructions and output format specifications. "
            "The optimized code should be in code block format:\n```cpp\n<code>\n```", 




# ===============================================================================================================================================================================================
"Operation_Command_Python": 
            "You will receive a Python code, which includes relevant I/O unit test cases, code functionality descriptions, and a Control Flow Graph (CFG) framework. "
            "Your task is to continue optimizing the code, ultimately providing the fastest code version.\n"
            "Please follow the steps below to optimize the code:\n"
            "1. Analyze the code functionality: Based on the I/O test cases and code functionality description, analyze the main functionality of the code.\n"
            "2. Generate fastest code: Combine the code functionality description and other information to generate the fastest code version. Please provide the most optimal and fastest code implementation.\n\n"
            "### Slow Code: \n```python\n{Slow_program}\n```\n\n"
            "### Test Case:"
            "### Code Functionality Description:\n```\n{Code_Function_Description}\n```\n\n"
            "### Control Flow Graph Framework of the Faster Version:\n```\n{Refer_Fast_CFG}\n```\n\n"
            "### Please follow the steps above and the output format to generate the fastest code version:\n",


            
"Operation_Command_Cpp": 
            "You will receive a C++ code, which includes relevant I/O unit test cases, code functionality descriptions, and a Control Flow Graph (CFG) framework. "
            "Your task is to continue optimizing the code, ultimately providing the fastest code version.\n"
            "Please follow the steps below to optimize the code:\n"
            "1. Analyze the code functionality: Based on the I/O test cases and code functionality description, analyze the main functionality of the code.\n"
            "2. Generate fastest code: Combine the code functionality description and other information to generate the fastest code version. Please provide the most optimal and fastest code implementation.\n\n"
            "### Slow Code: \n```cpp\n{Slow_program}\n```\n\n"
            "### Test Case:"
            "### Code Functionality Description:\n```\n{Code_Function_Description}\n```\n\n"
            "### Control Flow Graph Framework of the Faster Version:\n```\n{Refer_Fast_CFG}\n```\n\n"
            "### Please follow the steps above and the output format to generate the fastest code version:\n",



# ===============================================================================================================================================================================================
# IO_Test_Cot
"IO_Format": "Test Case {IO_ID}:\nInput: {IO_Input}\nOutput: {IO_Output}\n\n", 


}