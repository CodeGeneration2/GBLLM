Prompt_Dict = {


"Prompt_Description": "Code + NL_COT + CFG + I/O + Slow/Medium/Fast --> Efficient Code",


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
            "You will receive a Python code acceleration trajectory, which includes different versions of slow, medium, and fast code, relevant I/O unit test cases, code functionality descriptions, and a Control Flow Graph (CFG) framework. "
            "Your task is to continue optimizing the code along this acceleration trajectory, ultimately providing the fastest code version.\n"
            "Please follow the steps below to optimize the code:\n"
            "1. Analyze the code functionality: Based on the acceleration trajectory, I/O test cases, and code functionality description, analyze the main functionality of the code.\n"
            "2. Generate fastest code: Combine the code functionality description and other information to generate the fastest code version. Please provide the most optimal and fastest code implementation.\n\n"
            "### Code Acceleration Trajectory:\n"
            "Slow Code: \n```python\n{Slow_program}\n```\n\n"
            "Medium Code: \n```python\n{Medium_program}\n```\n\n"
            "Fast Code: \n```python\n{Fast_program}\n```\n\n"
            "### Test Case:"
            "### Code Functionality Description:\n```\n{Code_Function_Description}\n```\n\n"
            "### Control Flow Graph Framework of the Faster Version:\n```\n{Refer_Fast_CFG}\n```\n\n"
            "### Please follow the steps above and the output format to generate the fastest code version:\n",


            
"Operation_Command_Cpp": 
            "You will receive a C++ code acceleration trajectory, which includes different versions of slow, medium, and fast code, relevant I/O unit test cases, code functionality descriptions, and a Control Flow Graph (CFG) framework. "
            "Your task is to continue optimizing the code along this acceleration trajectory, ultimately providing the fastest code version.\n"
            "Please follow the steps below to optimize the code:\n"
            "1. Analyze the code functionality: Based on the acceleration trajectory, I/O test cases, and code functionality description, analyze the main functionality of the code.\n"
            "2. Generate fastest code: Combine the code functionality description and other information to generate the fastest code version. Please provide the most optimal and fastest code implementation.\n\n"
            "### Code Acceleration Trajectory:\n"
            "Slow Code: \n```cpp\n{Slow_program}\n```\n\n"
            "Medium Code: \n```cpp\n{Medium_program}\n```\n\n"
            "Fast Code: \n```cpp\n{Fast_program}\n```\n\n"
            "### Test Case:"
            "### Code Functionality Description:\n```\n{Code_Function_Description}\n```\n\n"
            "### Control Flow Graph Framework of the Faster Version:\n```\n{Refer_Fast_CFG}\n```\n\n"
            "### Please follow the steps above and the output format to generate the fastest code version:\n",



# ===============================================================================================================================================================================================
# IO_Test_Cot
"IO_Format": "Test Case {IO_ID}:\nInput: {IO_Input}\nOutput: {IO_Output}\n\n", 


}