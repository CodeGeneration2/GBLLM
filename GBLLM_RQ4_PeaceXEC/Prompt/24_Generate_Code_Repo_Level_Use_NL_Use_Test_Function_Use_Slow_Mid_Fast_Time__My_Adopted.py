Prompt_Dict = {


"Prompt_Description": "Code + NL_COT + CFG + I/O + Slow/Medium/Fast --> Efficient Code",


# ===============================================================================================================================================================================================
"Role_Play_Python": 
            "You are a software developer, and now you will help improve the execution speed of functions in the code. "
            "Please follow the instructions and output format specifications to generate a faster function. ", 
"Role_Play_Cpp": 
            "You are a software developer, and now you will help improve the execution speed of functions in the code. "
            "Please follow the instructions and output format specifications to generate a faster function. ", 






# ===============================================================================================================================================================================================
"Operation_Command_Python": 
            "You will be given an acceleration trajectory of a Python function, which includes different versions of slow, medium, and fast functions, related test cases, and a function functionality description. "
            "Your task is to continue optimizing the function along this acceleration trajectory, ultimately providing the fastest function version with an execution time within {Expected_Time} ms.\n"
            "Please follow the steps below to optimize the function:\n"
            "1. Analyze the function's functionality: Based on the acceleration trajectory, test cases, and the function's functionality description, analyze the main functionality of the function.\n"
            "2. Generate the fast function: Combine the function's functionality description and other information to generate the fastest function version with an execution time within {Expected_Time} ms. Please provide the most optimal and fastest function implementation.\n\n"
            "### Function Acceleration Trajectory:\n"
            "Slow function (Execution time {Slow_program_Time} ms): \n```python\n{Slow_program}\n```\n\n"
            "Medium function (Execution time {Medium_program_Time} ms): \n```python\n{Medium_program}\n```\n\n"
            "Fast function (Execution time {Fast_program_Time} ms): \n```python\n{Fast_program}\n```\n\n"
            "### Functionality Description:\n```\n{Code_Function_Description}\n```\n\n"
            "### Test Case:\n```python\n{Test_case}\n```\n\n"
            "### Please follow the steps above and the output format to gradually generate the fastest function version with an execution time within {Expected_Time} ms. "
            "The optimized function should use the function block format:\n```python\n{Code_Function_Head}\n<code>\n```\n",


            
"Operation_Command_Cpp": 
            "You will be given an acceleration trajectory of a C++ function, which includes different versions of slow, medium, and fast functions, related test cases, and a function functionality description. "
            "Your task is to continue optimizing the function along this acceleration trajectory, ultimately providing the fastest function version with an execution time within {Expected_Time} ms.\n"
            "Please follow the steps below to optimize the function:\n"
            "1. Analyze the function's functionality: Based on the acceleration trajectory, test cases, and the function's functionality description, analyze the main functionality of the function.\n"
            "2. Generate the fast function: Combine the function's functionality description and other information to generate the fastest function version with an execution time within {Expected_Time} ms. Please provide the most optimal and fastest function implementation.\n\n"
            "### Function Acceleration Trajectory:\n"
            "Slow function (Execution time {Slow_program_Time} ms): \n```cpp\n{Slow_program}\n```\n\n"
            "Medium function (Execution time {Medium_program_Time} ms): \n```cpp\n{Medium_program}\n```\n\n"
            "Fast function (Execution time {Fast_program_Time} ms): \n```cpp\n{Fast_program}\n```\n\n"
            "### Functionality Description:\n```\n{Code_Function_Description}\n```\n\n"
            "### Test Case:\n```cpp\n{Test_case}\n```\n\n"
            "### Please follow the steps above and the output format to gradually generate the fastest function version with an execution time within {Expected_Time} ms. "
            "The optimized function should use the function block format:\n```cpp\n{Code_Function_Head}\n<code>\n```\n",


# ===============================================================================================================================================================================================
# IO_Test_Cot
"IO_Format": "Test Case {IO_ID}:\nInput: {IO_Input}\nOutput: {IO_Output}\n\n", 



}