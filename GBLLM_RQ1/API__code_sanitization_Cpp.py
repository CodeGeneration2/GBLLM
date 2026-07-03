# -*- coding: utf-8 -*-
# print(f"### :\n{}")
# #####################################################################################################################🔖💡✅🟨


import tokenize
import re
import tokenize
import subprocess
import os
from io import StringIO
from transformers import AutoTokenizer



"""  Usage
from API__Cpp_code_clean import simple_cpp_code_cleaner
simple_cleaned_code = simple_cpp_code_cleaner(code)
from API__Cpp_code_clean import multi_cpp_code_cleaner
heavily_cleaned_code = multi_cpp_code_cleaner(code)
from API__Cpp_code_clean import knowledge_base_level_clean_cpp_func
cleaned_code = knowledge_base_level_clean_cpp_func(code)


"""



# #####################################################################################################################🔖💡✅🟨
DEBUG = False
SAMPLE_SOURCE_CODE = r"""
#include <iostream>  // For std::cin, std::cout, std::endl
#include <string>    // For std::string
#include <algorithm> // Not strictly necessary for this version, but often useful

int main() {
    // Optimize C++ standard streams for faster input/output.
    // This unties cin from cout and disables synchronization with C's stdio.
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int N;
    std::string S;
    std::cin >> N >> S;

    long long r = 0, g = 0, b = 0;
    // Count the occurrences of 'R', 'G', 'B' characters in the string.
    // This loop runs in O(N) time.
    for (char c : S) {
        if (c == 'R') {
            ++r;
        } else if (c == 'G') {
            ++g;
        } else { // c == 'B'
            ++b;
        }
    }

    // Calculate the total number of triplets (i, j, k) such that 0 <= i < j < k < N
    // and S[i], S[j], S[k] are all distinct ('R', 'G', 'B' in any order).
    // This is simply the product of the counts of each character.
    // This calculation is O(1) after initial counting.
    long long all_distinct_triplets = r * g * b;

    long long triplets_in_arithmetic_progression = 0;
    // Calculate the number of "bad" triplets: (i, j, k) where 0 <= i < j < k < N,
    // S[i], S[j], S[k] are all distinct, AND the indices form an arithmetic progression (j - i == k - j).
    // This part iterates through all possible pairs (i, j) with i < j,
    // and then derives k. This results in an O(N^2) time complexity.
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            // If S[i] and S[j] are the same, they cannot be part of a distinct triplet.
            if (S[i] == S[j]) {
                continue;
            }

            // Calculate the third index k such that i, j, k form an arithmetic progression.
            // j - i = k - j  =>  k = 2 * j - i
            int k = j * 2 - i;

            // Check if k is within the bounds of the string and if S[k] is distinct
            // from both S[i] and S[j].
            if (k < N && S[k] != S[i] && S[k] != S[j]) {
                // If all conditions are met, this is a "bad" triplet that needs to be subtracted.
                ++triplets_in_arithmetic_progression;
            }
        }
    }

    // The final answer is the total number of distinct triplets minus those that
    // also form an arithmetic progression.
    std::cout << all_distinct_triplets - triplets_in_arithmetic_progression << std::endl;

    return 0;
}
"""



# #####################################################################################################################🔖💡✅🟨
def main():
    # simple_clean_sample = simple_cpp_code_cleaner(SAMPLE_SOURCE_CODE)
    multi_clean_sample = simple_clean_and_remove_comments_cpp_func(SAMPLE_SOURCE_CODE)
    # clean_sample = knowledge_base_level_clean_cpp_func(SAMPLE_SOURCE_CODE)

    # print(f"✅✅✅ Source Code Simple Clean Sample:\n{simple_clean_sample}")
    print(f"✅✅✅ Source Code Multi Clean Sample:\n{multi_clean_sample}")



    



# #####################################################################################################################🔖💡✅🟨
def simple_cpp_code_cleaner(original_code):
    if original_code == "" or original_code is None or original_code == "pass":
        return "pass"
    source_code = original_code
    source_code = format_cpp_code_func(source_code)
    # Must repeat once, the previous step adds comments
    source_code = remove_empty_lines_func(source_code)

    return source_code





# #####################################################################################################################🔖💡✅🟨
def simple_clean_and_remove_comments_cpp_func(original_code):
    if original_code == "" or original_code is None or original_code == "pass":
        return "pass"

    source_code = original_code
    source_code = remove_comments_func(source_code)
    source_code = format_cpp_code_func(source_code)
    # Must repeat once, the previous step adds comments
    source_code = remove_comments_func(source_code)
    source_code = remove_empty_lines_func(source_code)

    return source_code




# #####################################################################################################################🔖💡✅🟨
def initialize_vocab_func(DeepSeek_V3_vocab_path=r"E:\Python_Parameters\DeepSeek_V3_tokenizer"):
    # DeepSeek_V3_vocab_path = r"DeepSeek_V3_tokenizer"
    tokenizer_vocab = AutoTokenizer.from_pretrained( DeepSeek_V3_vocab_path )# trust_remote_code=True
    return tokenizer_vocab


# #####################################################################################################################🔖💡✅🟨
def knowledge_base_level_clean_cpp_func(original_code, tokenizer_vocab):
    if original_code == "" or original_code is None or original_code == "pass":
        return "pass"

    source_code = original_code
    source_code = remove_comments_func(source_code)
    source_code = format_cpp_code_func(source_code)
    # Must repeat once, the previous step adds comments
    source_code = remove_comments_func(source_code)

    """
    source_code = handle_ast_replacement_func(source_code)
    source_code = ast_clean_func(source_code)
    source_code = remove_recursion_limit_func(source_code.strip())  
    source_code = remove_main_func(source_code)
    source_code = replace_is_func(source_code)
    source_code = benign_replacement_func(source_code)
    source_code = replace_strange_words_func(source_code)
    source_code = remove_logging_func(source_code)
    source_code = remove_new_process_func(source_code)
    source_code = replace_strange_io_func(source_code)
    source_code = remove_exit_func(source_code)
    source_code = remove_exec_func(source_code)
    source_code = remove_inline_newlines_func(source_code)
    """
    
    source_code = remove_empty_lines_func(source_code)
    source_code = remove_overly_long_code_func(source_code, tokenizer_vocab, max_length=1024)


    return source_code




# #####################################################################################################################🔖💡✅🟨
def format_cpp_code_func(source_code) -> str:
    # Temporary file
    with open("temp_format_Cpp.cpp", 'w', encoding='utf-8') as f:
        f.write(source_code)

    # Call clang-format to format code
    result = subprocess.run(
        ['clang-format', "temp_format_Cpp.cpp"],
        capture_output=True,
        text=True,
        encoding='utf-8'  # Force use of UTF-8 encoding
    )

    # Return formatted code
    formatted_cpp_code = result.stdout

    os.remove("temp_format_Cpp.cpp")  # Delete temporary file

    return formatted_cpp_code




# #####################################################################################################################🔖💡✅🟨
# Derived from CodeBLEU: https://github.com/k4black/codebleu
def remove_comments_func(code_str, lang='cpp'):
    if lang in ["python"]:
        """
        Returns 'source' minus comments and docstrings.
        """
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
def remove_overly_long_code_func(code_str, tokenizer_vocab, max_length=1024):


    token_encoding = tokenizer_vocab.encode(code_str.strip())

    if len(token_encoding) > max_length:
        return "pass"
    else:
        return code_str.strip()




# #####################################################################################################################🔖💡✅🟨
if __name__ == "__main__":
    main()