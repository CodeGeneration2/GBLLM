# -*- coding: utf-8 -*-
# print(f"### :\n{}")
# #####################################################################################################################🔖💡✅🟨

import pandas as pd
import os
from tqdm import tqdm
import ast, astor
import io
import tokenize
import keyword
import re
from io import StringIO

""" Usage
from API__code_clean_Python__latest5 import simple_python_code_cleaner
from API__code_clean_Python__latest5 import simple_clean_and_remove_comments_func
from API__code_clean_Python__latest5 import heavy_python_code_cleaner

simple_cleaned_code = simple_python_code_cleaner(code)
simple_cleaned_code = simple_clean_and_remove_comments_func(code)
heavily_cleaned_code = heavy_python_code_cleaner(code)
"""

# #####################################################################################################################🔖💡✅🟨
INLINE_NEWLINE_KEYWORDS = ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 
'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'int', 'float']

# #####################################################################################################################🔖💡✅🟨
DEBUG = False
SAMPLE_SOURCE_CODE = """
import os
import sys
import numpy as np
from numba.pycc import CC
from my_module import solve
def function1(var1, var2, var3):
    var4 = np.full(var1 + 1, 10 ** 18, dtype=np.int64)
    for var6 in range(1 << var1):
        var11 = (var6 & 21845) + (var6 >> 1 & 21845)
        var11 = (var11 & 13107) + (var11 >> 2 & 13107)
        var11 = (var11 & 3855) + (var11 >> 4 & 3855)
        var11 = (var11 & 255) + (var11 >> 8 & 255)
        var5 = var6
        while var5:
            var4[var11] = min(var4[var11], function2(var5, var6))
            var5 = var5 - 1 & var6
        var4[var11] = min(var4[var11], function2(0, var6))
    return var4
def function2(var5, var6):
    var7 = 0
    var8 = var2[var5]
    var9 = var3[var5 ^ var6]
    for var10 in range(var1):
        var7 += min(var8[var10], var9[var10])
    return var7
if sys.argv[-1] == 'ONLINE_JUDGE':
    var12 = CC('my_module')
    var12.export('solve', '(i8, i8[:,:], i8[:,:])')(var13)
    var12.compile()
    exit()

var14 = np.fromstring(sys.stdin.read(), dtype=np.int64, sep=' ')
var1 = var14[0]
var15 = var14[1::3]
var16 = var14[2::3]
var17 = var14[3::3]
var18 = (np.arange(1 << var1)[:, None] & 1 << np.arange(var1) > 0).astype(np.int64)
var2 = abs((var15[None, :] * var18)[..., None] - var15[None, None, :]).min(axis =1) * var17[None, :]
var3 = abs((var16[None, :] * var18)[..., None] - var16[None, None, :]).min(axis =1) * var17[None, :]
var4 = var13(var1, var2, var3)
print('\n'.join(map(str, var4)))
"""

# #####################################################################################################################🔖💡✅🟨
def main():
    simple_clean_sample = simple_python_code_cleaner(SAMPLE_SOURCE_CODE)
    simple_clean_no_comments_sample = simple_clean_and_remove_comments_func(SAMPLE_SOURCE_CODE)
    heavy_clean_sample = heavy_python_code_cleaner(SAMPLE_SOURCE_CODE)
    print(f"### Source Code Heavy Clean Sample:\n{heavy_clean_sample}")

# #####################################################################################################################🔖💡✅🟨
def simple_python_code_cleaner(original_code):
    source_code = original_code
    
    source_code = remove_recursion_limit_func(source_code.strip())  
    source_code = remove_main_block_func(source_code)
    source_code = replace_is_func(source_code)
    source_code = benign_replacement_func(source_code)
    source_code = remove_inline_newlines_func(source_code)
    source_code = remove_empty_lines_func(source_code)

    return source_code

# #####################################################################################################################🔖💡✅🟨
def simple_clean_and_remove_comments_func(original_code):
    source_code = original_code
    
    source_code = handle_ast_replacement_func(source_code)
    source_code = ast_clean_func(source_code)
    source_code = remove_multiline_comments_func(source_code)
    source_code = remove_recursion_limit_func(source_code.strip())  
    source_code = remove_main_block_func(source_code)
    source_code = replace_is_func(source_code)
    source_code = benign_replacement_func(source_code)

    # Specific to heavy cleaning (commented out in this function)
    # source_code = replace_strange_words_func(source_code)
    # source_code = remove_logging_func(source_code)
    # source_code = remove_multiprocessing_func(source_code)
    # source_code = replace_strange_io_func(source_code)
    # source_code = remove_exit_func(source_code)
    # source_code = remove_exec_func(source_code)

    source_code = remove_inline_newlines_func(source_code)
    source_code = remove_empty_lines_func(source_code)

    return source_code

# #####################################################################################################################🔖💡✅🟨
def heavy_python_code_cleaner(original_code):
    source_code = original_code

    source_code = handle_ast_replacement_func(source_code)
    source_code = ast_clean_func(source_code)
    source_code = remove_multiline_comments_func(source_code)
    source_code = remove_recursion_limit_func(source_code.strip())  
    source_code = remove_main_block_func(source_code)
    source_code = replace_is_func(source_code)
    source_code = benign_replacement_func(source_code)

    # Specific to heavy cleaning
    source_code = replace_strange_words_func(source_code)
    source_code = remove_logging_func(source_code)
    source_code = remove_multiprocessing_func(source_code)
    source_code = replace_strange_io_func(source_code)
    source_code = remove_exit_func(source_code)
    source_code = remove_exec_func(source_code)

    source_code = remove_inline_newlines_func(source_code)
    source_code = remove_empty_lines_func(source_code)

    return source_code

# #####################################################################################################################🔖💡✅🟨
def handle_ast_replacement_func(code_str): 
    """Pre-processing special cases"""
    # print(f"### handle_ast_replacement_func, code_str:\n{code_str}")
    code_str = code_str.replace("print((*", "print(tuple(")
    code_str = code_str.replace("ÈëÁ¦Àý", "")
    code_str = code_str.replace("¤»¤¤¤¬¤ï", "fun")
    code_str = code_str.replace("¤Õ¤¬¤ï", "func")
    code_str = code_str.replace("sudo reboot", "")
    code_str = code_str.replace("shutdown -r", "")
    code_str = code_str.replace("shutdown ", "")
    code_str = code_str.replace('if os.name == "posix":\n    ', 'if os.name == "posix":\n    pass\n    ')

    return code_str.strip()

# #####################################################################################################################🔖💡✅🟨
def ast_clean_func(code_string):
    """ First convert to AST, then convert back """
    try:
        ast_model = ast.parse(code_string.strip())
        cleaned_code = astor.to_source(ast_model)
        return cleaned_code.strip()
    except:
        if DEBUG:
            print(f"### AST Failed, code string:\n{code_string}")
        return code_string

# #####################################################################################################################🔖💡✅🟨
class CommentRemover(ast.NodeTransformer) :
    def visit(self, node):
        # Remove comment nodes
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
            return None
        else:
            return ast.NodeTransformer.visit(self, node)
        
# #####################################################################################################################🔖💡✅🟨
# Derived from CodeBLEU: https://github.com/k4black/codebleu
def remove_multiline_comments_func(code_str, lang='python'):
    """ Returns 'source code' minus comments and documentation. """
    def _remove_multiline_comments_inner():
        # ------------------------------------------
        # Remove multiline comments, keep only code, applicable to Python
        if lang in ["python"]:
            io_obj = StringIO(code_str)
            out = ""
            prev_toktype = tokenize.INDENT
            last_lineno = -1
            last_col = 0
            for tok in tokenize.generate_tokens(io_obj.readline):
                token_type = tok[0]
                token_string = tok[1]
                start_line, start_col = tok[2]
                end_line, end_col = tok[3]
                tok[4]
                if start_line > last_lineno:
                    last_col = 0
                if start_col > last_col:
                    out += " " * (start_col - last_col)
                # Remove comments:
                if token_type == tokenize.COMMENT:
                    pass
                # This series of conditionals removes docstrings:
                elif token_type == tokenize.STRING:
                    if prev_toktype != tokenize.INDENT:
                        # This is likely a docstring; double-check we're not inside an operator:
                        if prev_toktype != tokenize.NEWLINE:
                            if start_col > 0:
                                out += token_string
                else:
                    out += token_string
                prev_toktype = token_type
                last_col = end_col
                last_lineno = end_line
            temp = []
            for x in out.split("\n"):
                if x.strip() != "":
                    temp.append(x)
            return "\n".join(temp)
        elif lang in ["ruby"]:
            return code_str
        else:
            def replacer(match):
                s = match.group(0)
                if s.startswith("/"):
                    return " "  # note: a space and not an empty string
                else:
                    return s

            pattern = re.compile(
                r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                re.DOTALL | re.MULTILINE,
            )
            temp = []
            for x in re.sub(pattern, replacer, code_str).split("\n"):
                if x.strip() != "":
                    temp.append(x)
            return "\n".join(temp)
        
    # Validation check
    try:
        return _remove_multiline_comments_inner()
    except:
        return code_str

# #####################################################################################################################🔖💡✅🟨
def remove_recursion_limit_func(code):
    if "setrecursionlimit(" not in code and "stack_size(" not in code:
        return code
    
    # ------------------------------------------
    code_lines = code.split("\n")
    new_code_list = []
    for line in code_lines:
        if "setrecursionlimit(" in line or "stack_size(" in line:
            if DEBUG:
                print(f"### setrecursionlimit( or stack_size(:\n{line}")
        else:
            new_code_list.append(line)
    cleaned_code = "\n".join(new_code_list)

    return cleaned_code.strip()

# #####################################################################################################################🔖💡✅🟨
def remove_main_block_func(code):
    if "__name__" not in code or "__main__" not in code:
        return code
    
    try:
        ast_model = ast.parse(code.strip())
        for node_index, node in enumerate(ast_model.body):
            # --------------------------------------- If it is a function body variable declaration ------------------#
            if isinstance(node, ast.If):
                try:
                    if node.test.left.id == '__name__' and node.test.comparators[0].value == '__main__':
                        ast_model.body[node_index:node_index + 1] = node.body
                except: # No node info
                    continue

        cleaned_code = astor.to_source(ast_model)
        return cleaned_code.strip()
    
    except:
        return code
    
# #####################################################################################################################🔖💡✅🟨
def replace_is_func(code): 
    if " is not " not in code and " is " not in code:
        return code
    else:
        code = code.replace(" is not ", " != ")
        code = code.replace(" is ", " == ")
        return code.strip()

# #####################################################################################################################🔖💡✅🟨
def benign_replacement_func(code_str): 

    # ✅
    for match_item in ["sys.stdin.readline", "stdin.readline", "sys.stdin.buffer.readline", "stdin.buffer.readline"]:
        if match_item in code_str:
            code_str = code_str.replace(match_item, "input") 

    # ✅
    for match_item in ["sys.stdout.write", "stdout.write", "sys.__stdout__.write", "__stdout__.write", "sys.stderr.write", "stderr.write", "sys.__stderr__.write", "__stderr__.write"]:
        if match_item in code_str:
            code_str = code_str.replace(match_item, "print") 

    # ✅
    if 'return sys.stdout.flush()' in code_str:
        code_str = code_str.replace('return sys.stdout.flush()', "return") 
    if 'return stdout.flush()' in code_str:
        code_str = code_str.replace('return stdout.flush()', "return") 

    return code_str.strip()

# #####################################################################################################################🔖💡✅🟨
def replace_strange_words_func(original_code_str): 
    code_str = original_code_str

    # ✅=======================================================================================================================================================
    def replace_import(code_str):
        code_lines_list = code_str.split("\n")
        new_code_lines_list = []
        for code_line in code_lines_list:
            if "import" in code_line and ("stdout" in code_line or "stderr" in code_line):
                if DEBUG:
                    print(f"### Before modification：{code_line}")
                code_line = code_line.replace("from sys import stdout, ", "from sys import ")
                code_line = code_line.replace(", stdout", "")
                code_line = code_line.replace("from sys import stdout", "")
                code_line = code_line.replace("print = __import__('sys').stdout.write", "")
                code_line = code_line.replace("from sys import stderr, ", "from sys import ")
                code_line = code_line.replace(", stderr", "")
                code_line = code_line.replace("from sys import stderr", "")
                code_line = code_line.replace("print = __import__('sys').stderr.write", "")
                if code_line.strip() != "":
                    new_code_lines_list.append(code_line)
                if DEBUG:
                    print(f"### After modification：{code_line}")
                    if "stdout" in code_line or "stderr" in code_line:
                        print(f"### stdout in or stderr in code_line:\n{code_line}")
            else:
                new_code_lines_list.append(code_line)
        
        code_str = "\n".join(new_code_lines_list)
        code_str = code_str.strip()
        return code_str

    if "import" in code_str and ("stdout" in code_str or "stderr" in code_str):
        if DEBUG:
            print(f"### Code path：{code_path}")
        code_str = replace_import(code_str)
        

    # ✅=======================================================================================================================================================
    for match_item in [", file=sys.stdout", ", file=stdout", ", output=sys.stdout", ", output=stdout", ", file=sys.stderr", ", file=stderr", ", output=sys.stderr", ", output=stderr"]:
        if match_item in code_str:
            code_str = code_str.replace(match_item, "")

    # ✅=======================================================================================================================================================
    def replace_strange_io(code_str):
        code_lines_list2 = code_str.split("\n")
        new_code_lines_list2 = []
        for code_line in code_lines_list2:

            # Check if the line of code contains any of the matching items
            # if any(match_item in code_line for match_item in ["IOWrapper", "FastIO", "StringIO", "BytesIO", "FastStdout", "stdout", "stderr", "open('", ".close("]):
            if 'class ' not in code_line and any(match_item in code_line for match_item in ["IOWrapper(", "FastIO(", "StringIO(", "BytesIO(", "FastStdout(", ".close(", "stdout", "stderr"]):
                indent_count = len(code_line) - len(code_line.lstrip())
                indent_string = code_line[:indent_count]
                new_code_lines_list2.append(f'{indent_string}pass')
                continue  # If match item is in the code line, skip that line
            else:
                new_code_lines_list2.append(code_line)  # Otherwise add the line to the new list
        
        code_str = "\n".join(new_code_lines_list2)
        code_str = code_str.strip()
        return code_str

    if any(match_item in code_str for match_item in ["IOWrapper(", "FastIO(", "StringIO(", "BytesIO(", "FastStdout(", ".close(", "stdout", "stderr"]):
        code_str = replace_strange_io(code_str)

    # ✅=======================================================================================================================================================
    def replace_open(code_str):
        code_lines_list3 = code_str.split("\n")
        new_code_lines_list3 = []
        for code_line in code_lines_list3:
            if 'open(0' in code_line:
                new_code_lines_list3.append(code_line)
                continue
            elif 'open(' in code_line:
                indent_count = len(code_line) - len(code_line.lstrip())
                indent_string = code_line[:indent_count]
                new_code_lines_list3.append(f'{indent_string}pass')
                continue
            else:
                new_code_lines_list3.append(code_line)

        code_str = "\n".join(new_code_lines_list3)
        code_str = code_str.strip()

        return code_str

    if 'open(' in code_str:
        code_str = replace_open(code_str)

    # =========================================================================================
    return code_str.strip()

# #####################################################################################################################🔖💡✅🟨
def remove_logging_func(code_str): 
    """ Remove logging related code lines """
    if not (any(match_item in code_str for match_item in ["unittest.", "traceback.", "os.write", "logging.", "os.system", "import division", "__future__"])):
        return code_str

    # ------------------------------------------
    code_lines_list2 = code_str.split("\n")
    new_code_lines_list2 = []
    for code_line in code_lines_list2:

        # Check if the line of code contains any of the matching items
        if 'class ' not in code_line and any(match_item in code_line for match_item in ["Unittest.", "traceback.", "os.write", "logging.", "os.system", "import division", "__future__"]):
            indent_count = len(code_line) - len(code_line.lstrip())
            indent_string = code_line[:indent_count]
            new_code_lines_list2.append(f'{indent_string}pass')
            if DEBUG:
                print(f"### Delete logging, code line：{code_line}")
            continue  # If match item is in the code line, skip that line
        else:
            new_code_lines_list2.append(code_line)  # Otherwise add the line to the new list
    
    code_str = "\n".join(new_code_lines_list2)
    code_str = code_str.strip()

    return code_str

# #####################################################################################################################🔖💡✅🟨
def remove_multiprocessing_func(code_str): 

    # ✅=========================================================================================
    def replace_import(code_str):
        code_lines_list = code_str.split("\n")
        new_code_lines_list = []
        for code_line in code_lines_list:
            if "import" in code_line and "threading" in code_line:            
                code_line = code_line.replace("import threading, ", "import ")
                code_line = code_line.replace(", threading", "")
                code_line = code_line.replace("import threading", "")
                if code_line.strip() != "":
                    new_code_lines_list.append(code_line)
            else:
                new_code_lines_list.append(code_line)

        code_str = "\n".join(new_code_lines_list)
        code_str = code_str.strip()
        return code_str

    if "import" in code_str and "threading" in code_str:
        code_str = replace_import(code_str)


    # ✅=========================================================================================
    def replace_threading(code_str):
        code_lines_list = code_str.split("\n")
        new_code_lines_list = []
        for code_line in code_lines_list:
            if "threading.Thread(target=" in code_line:            
                function_name = code_line.split("threading.Thread(target=")[1].split(")")[0]
                new_code_lines_list.append(f'{function_name}()')
            else:
                new_code_lines_list.append(code_line)

        code_str = "\n".join(new_code_lines_list)
        code_str = code_str.strip()
        return code_str

    if 'threading.Thread(target=' in code_str:
        code_str = replace_threading(code_str)


    # ✅=========================================================================================
    def replace_start(code_str):
        code_lines_list = code_str.split("\n")
        new_code_lines_list = []
        for code_line in code_lines_list:
            if ".start()" in code_line or ".join()" in code_line:           
                indent_count = len(code_line) - len(code_line.lstrip())
                indent_string = code_line[:indent_count]
                new_code_lines_list.append(f'{indent_string}pass')  
                continue
            else:
                new_code_lines_list.append(code_line)

        code_str = "\n".join(new_code_lines_list)
        code_str = code_str.strip()
        return code_str

    if ".start()" in code_str or ".join()" in code_str:
        code_str = replace_start(code_str)


    # ✅=========================================================================================
    def remove_all_remaining(code_str):
        code_lines_list = code_str.split("\n")
        new_code_lines_list = []
        for code_line in code_lines_list:
            if any(match_item in code_line for match_item in ["threading", "thread", "multiprocessing", "asyncio", "queue.Queue(", "ProcessPoolExecutor", "concurrent", "fork(", "subprocess.run("]):        
                indent_count = len(code_line) - len(code_line.lstrip())
                indent_string = code_line[:indent_count]
                new_code_lines_list.append(f'{indent_string}pass')  
                if DEBUG:
                    print(f"### Remove all remaining, code line:\n{code_line}")
                continue
            else:
                new_code_lines_list.append(code_line)

        code_str = "\n".join(new_code_lines_list)
        code_str = code_str.strip()
        return code_str
    

    if any(match_item in code_str for match_item in ["threading", "thread", "multiprocessing", "asyncio", "queue.Queue(", "ProcessPoolExecutor", "concurrent", "fork(", "subprocess.run("]):
        code_str = remove_all_remaining(code_str)


    return code_str.strip()

# #####################################################################################################################🔖💡✅🟨
def replace_strange_io_func(code_str):
    if not (any(match_item in code_str for match_item in ["IOWrapper", "FastIO", "StringIO", "BytesIO", "FastStdout", "IOBase", "setrecursionlimit"])):
        return code_str

    code_lines_list2 = code_str.split("\n")
    new_code_lines_list2 = []
    for code_line in code_lines_list2:
        # Check if the line of code contains any of the matching items
        if any(match_item in code_line for match_item in ["IOWrapper", "FastIO", "StringIO", "BytesIO", "FastStdout", "IOBase", "setrecursionlimit"]):
            indent_count = len(code_line) - len(code_line.lstrip())
            indent_string = code_line[:indent_count]
            new_code_lines_list2.append(f'{indent_string}pass')
            if DEBUG:
                print(f"### Replace strange IO, code line:\n{code_line}")
            continue  # If match item is in the code line, skip that line
        else:
            new_code_lines_list2.append(code_line)  # Otherwise add the line to the new list
    
    code_str = "\n".join(new_code_lines_list2)
    code_str = code_str.strip()

    return code_str

# #####################################################################################################################🔖💡✅🟨
def remove_exit_func(original_code_str): 

    code_str = original_code_str

    # ✅=======================================================================================================================================================
    def replace_exit(code_str):
        code_lines_list3 = code_str.split("\n")
        new_code_lines_list3 = []
        for code_line in code_lines_list3:
            if 'exit(' in code_line or 'quit(' in code_line:
                stripped_code_line = code_line.lstrip()
                indent_count = len(code_line) - len(stripped_code_line)
                indent_string = code_line[:indent_count]
                new_code_lines_list3.append(f"{indent_string}return")
            else:
                new_code_lines_list3.append(code_line)

        code_str = "\n".join(new_code_lines_list3)
        code_str = code_str.strip()

        return code_str

    if 'exit(' in code_str or 'quit(' in code_str:
        code_str = replace_exit(code_str)

    return code_str.strip()

# #####################################################################################################################🔖💡✅🟨
def remove_exec_func(code_str):
    if "exec(" not in code_str:
        return code_str

    code_lines_list2 = code_str.split("\n")
    new_code_lines_list2 = []
    for code_line in code_lines_list2:
        # Check if the line of code contains any of the matching items
        if "exec(" in code_line:
            indent_count = len(code_line) - len(code_line.lstrip())
            indent_string = code_line[:indent_count]
            new_code_lines_list2.append(f'{indent_string}pass')  
            continue  # If match item is in the code line, skip that line
        else:
            new_code_lines_list2.append(code_line)  # Otherwise add the line to the new list
    
    code_str = "\n".join(new_code_lines_list2)
    code_str = code_str.strip()

    return code_str

# ##############################################################################################################################
def remove_inline_newlines_func(source_code: str) -> str:
    source_code_line_slice = source_code.strip().split('\n')
    
    tokens_group_list = tokenize.generate_tokens(io.StringIO(source_code.strip()).readline)

    if tokens_group_list == []:
        return 'pass'

    deletion_list = []
    try:
        for tok_type, tok_string, start, end, line in tokens_group_list:
            if tok_type == tokenize.NL:
                if start[1] > 5:
                    deletion_list.append(int(start[0]) - 1)
    except:
        if DEBUG:
            print(f"### remove_inline_newlines_func Error：")
        return source_code

    if len(deletion_list) == 0:
        return source_code.strip()
    
    deletion_list.reverse()

    for line_number in deletion_list:
        front_part = source_code_line_slice[line_number].rstrip()
        try:
            back_part = source_code_line_slice[line_number + 1].lstrip()
        except:
            if DEBUG:
                print(f"### remove_inline_newlines_func Error")
            return source_code.strip()
        # print(f"front_part: {front_part[-5:]}    \t  back_part: {back_part[:5]}")
        add_space = False
        for keyword in INLINE_NEWLINE_KEYWORDS:
            if front_part.endswith(keyword) or back_part.startswith(keyword):
                add_space = True
                break
        
        if add_space:
            source_code_line_slice[line_number] = source_code_line_slice[line_number].rstrip() + ' ' + source_code_line_slice[line_number + 1].lstrip()
        else:
            source_code_line_slice[line_number] = source_code_line_slice[line_number].rstrip() + source_code_line_slice[line_number + 1].lstrip()
        
        # Delete the original next element
        del source_code_line_slice[line_number + 1]

    returned_source_code = '\n'.join(source_code_line_slice)
    
    return returned_source_code.strip()

# #####################################################################################################################🔖💡✅🟨
def remove_empty_lines_func(code_str):
    code_lines_list = code_str.split("\n")
    new_code_lines_list = []
    for code_line in code_lines_list:
        if code_line.strip() != "":
            new_code_lines_list.append(code_line)
    code_str = "\n".join(new_code_lines_list)
    
    return code_str.strip()

# #####################################################################################################################🔖💡✅🟨
if __name__ == "__main__":
    main()