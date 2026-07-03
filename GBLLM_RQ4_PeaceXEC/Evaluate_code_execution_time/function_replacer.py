import re
import os
import json
import ast
import astor
import logging


# #####################################################################################################################🔖💡🟨✓✗
DEBUG = False


# #####################################################################################################################🔖💡🟨 ✅✅✅ ❌❌❌
def replace_function_with_llm_generated_code(target_code_path, target_class_name, target_func_name, new_signature_and_body):
    """
    Replaces the function signature and body in a Python file based on the file path, class name, and function name,
    while maintaining consistent indentation.

    Args:
        target_code_path (str): The path to the Python file.
        target_class_name (str): The class name. If None, the function is searched for in the global scope.
        target_func_name (str): The name of the function to replace.
        new_signature_and_body (str): The new function signature and body.

    Returns:
        int: Returns 1 if replacement is successful, otherwise returns 0.
    """
    try:
        # Read file content
        with open(target_code_path, 'r', encoding='utf-8') as f:
            code_lines = f.readlines()

        # Initialize variables
        inside_target_function = False
        function_start_line = None
        function_end_line = None
        indent_level = None
        class_indent = None

        # Compile regex pattern to match function definition
        function_match_pattern = re.compile(r'^\s*def\s+' + re.escape(target_func_name) + r'\s*\(.*\)\s*')

        # Iterate through each line in the file
        for line_idx, line_code in enumerate(code_lines):
            if target_class_name:
                if f"class {target_class_name}" in line_code:
                    class_indent = len(line_code) - len(line_code.lstrip())
                    inside_target_function = False
                elif inside_target_function:
                    if line_code.strip() == '' or len(line_code) - len(line_code.lstrip()) <= indent_level:
                        function_end_line = line_idx
                        break

            if function_match_pattern.match(line_code):
                if target_class_name:
                    if (len(line_code) - len(line_code.lstrip())) == class_indent + 4:
                        function_start_line = line_idx
                        indent_level = len(line_code) - len(line_code.lstrip())
                        inside_target_function = True
                        continue
                else:
                    function_start_line = line_idx
                    indent_level = len(line_code) - len(line_code.lstrip())
                    inside_target_function = True
                    continue

        # If function start line is found, continue to find the end line
        if inside_target_function and function_start_line is not None:
            for line_idx in range(function_start_line + 1, len(code_lines)):
                line_code = code_lines[line_idx]
                if len(line_code) - len(line_code.lstrip()) <= indent_level:
                    function_end_line = line_idx
                    break

        # If both start and end lines are found, perform replacement
        if function_start_line is not None and function_end_line is not None:
            indentation = ' ' * indent_level
            new_function_lines = [indentation + line_code + "\n" if line_code.strip() else line_code for line_code in new_signature_and_body.split('\n')]

            
            print(f"\n### Before: {code_lines[:function_start_line][-50:]}")
            print(f"\n### After: {code_lines[function_end_line:][:50]}")


            code_lines = code_lines[:function_start_line] + new_function_lines + code_lines[function_end_line:]

            print(f"\n### Actual: {code_lines[function_start_line:function_start_line+20]}")

            print('\n'.join(new_function_lines))




            # Write modified content back to file
            with open(target_code_path, 'w', encoding='utf-8') as f:
                f.writelines(code_lines)

            print(f"### Function '{target_func_name}' successfully replaced in file '{target_code_path}'.")
            return 1
        else:
            print(f"### Function '{target_func_name}' not found in file '{target_code_path}'.")
            return 0
    except Exception as e:
        print(f"### Error occurred while replacing function in file '{target_code_path}': {e}")
        return 0




# ####################################################################################################################🔖💡🟨 ✅✅✅ ❌❌❌ 
def replace_target_file_function_body_with_source(target_code_path, target_func_name, new_function_body, target_class_name=None):
    """
    Replaces the body of a specified function in the target file with the function body from the source.
    
    Parameters
    ----------
    target_code_path : str
        The path to the target file where the function body needs to be replaced.
    target_func_name : str
        The name of the function whose body needs to be replaced.
    new_function_body : str
        The source string containing the new function body.
    target_class_name : str, optional
        The class name to look for methods in, if applicable.
        
    Returns
    -------
    bool
        Returns True if replacement is successful, otherwise False.
    """

    try:
        # Read new function body from source file (Commented out in original)
        # with open(new_function_body, 'r', encoding='utf-8') as file:
        #     new_function_body = file.read()

        # Read target file source code
        with open(target_code_path, 'r', encoding='utf-8') as file:
            original_code = file.read()

        # Parse target file source code
        original_code_ast = ast.parse(original_code)

        # Helper function to replace function body
        def replace_body_helper(node):
            for subnode in ast.iter_child_nodes(node):
                if isinstance(subnode, ast.FunctionDef) and subnode.name == target_func_name:
                    new_code_ast = ast.parse(new_function_body).body
                    # assert len(new_code_ast) == 1, "### This function body code should contain only one function definition"
                    # assert subnode.name == new_code_ast[-1].name, "### Function name mismatch, cannot replace"
                    subnode.body = new_code_ast[-1].body
                    if DEBUG:
                        print(f"\n✅✅✅ Successfully replaced: Replaced body of function '{target_func_name}' in {'class ' + target_class_name if target_class_name else 'global scope'}.")
                    return True
            return False

        # Execute replacement in specified class or global scope
        if (target_class_name) and (target_class_name != "Null") and (target_class_name is not None):
            for node in ast.walk(original_code_ast):
                if isinstance(node, ast.ClassDef) and node.name == target_class_name:
                    if replace_body_helper(node):
                        break
            else:
                print(f"❌❌❌ Error: Class '{target_class_name}' not found in {target_code_path}.")
                return False
        else:
            if not replace_body_helper(original_code_ast):
                print(f"❌❌❌ Function '{target_func_name}' not found in {target_code_path}.")
                return False

        # Write modified AST back to target file
        with open(target_code_path, 'w', encoding='utf-8') as file:
            file.write(astor.to_source(original_code_ast))
        
        return True

    except FileNotFoundError:
        print(f"❌❌❌ One of the files was not found: {new_function_body} or {target_code_path}.")
        return False
    except Exception as e:
        print(f"❌❌❌ Failed to replace code: Error occurred while replacing function body: {e}")
        return False




# #####################################################################################################################🔖💡✅🟨
# after_code = """
# def lines_with_leading_tabs_expanded(s):
#     lines = []
#     append_line = lines.append
#     split_lines = s.splitlines()
#     for line in split_lines:
#         stripped_line = line.lstrip()
#         if not stripped_line or stripped_line == line:
#             append_line(line)
#         else:
#             prefix_length = len(line) - len(stripped_line)
#             prefix = line[:prefix_length].expandtabs()
#             append_line(prefix + stripped_line)
#     if s.endswith(""):
#         append_line("")
#     return lines
# """
if __name__ == "__main__":
    # file_path = r"E:\MEGA\peace\docker\repo_python\black\src\black\strings.py"
    # target_class_name = None
    # target_func_name = "lines_with_leading_tabs_expanded"
    # new_signature_and_body = after_code
    # replace_target_file_function_body_with_source(file_path, target_func_name, new_signature_and_body, target_class_name)



    # Example usage
    replace_function_with_llm_generated_code("your_file_path.py", "", "your_function_name", """
def your_function_name():
    # Function body
    pass
""")