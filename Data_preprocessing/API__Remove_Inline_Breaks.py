import pandas as pd
import os
from tqdm import tqdm
import ast
import astor
import io
import tokenize
import keyword

# Keywords that require a space when merging inline-broken lines
INLINE_BREAK_KEYWORDS = [
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
    'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally',
    'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal',
    'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'int', 'float'
]

DEBUG = False

# Example source code to test cleaning
sample_source_code = """
def main():
    pass
"""

def remove_inline_line_breaks(source: str) -> str:
    """
    Merge lines that were split inline by tokenize.NL tokens.
    If the break occurs after a keyword or before a keyword, insert a space.
    """
    # Split source into individual lines
    lines = source.strip().split('\n')
    # Generate tokens to find inline breaks
    tokens = tokenize.generate_tokens(io.StringIO(source.strip()).readline)

    # Collect the line indices where an inline break occurs
    drop_indices = []
    for tok_type, tok_string, start, end, line in tokens:
        # tokenize.NL signals a non-logical newline; check if it's indented >5 columns
        if tok_type == tokenize.NL and start[1] > 5:
            drop_indices.append(start[0] - 1)

    # If there are no inline breaks to fix, return the stripped source
    if not drop_indices:
        return source.strip()

    # Process breaks from last to first to avoid index shifts
    drop_indices.reverse()

    for idx in drop_indices:
        head = lines[idx].rstrip()      # part before the break
        tail = lines[idx + 1].lstrip()  # part after the break

        # Determine if a space is needed between merged segments
        space_needed = False
        for kw in INLINE_BREAK_KEYWORDS:
            if head.endswith(kw) or tail.startswith(kw):
                space_needed = True
                break

        # Merge with or without a space
        if space_needed:
            lines[idx] = f"{head} {tail}"
        else:
            lines[idx] = f"{head}{tail}"

        # Remove the now-redundant following line
        del lines[idx + 1]

    # Reassemble and return the cleaned code
    result = '\n'.join(lines)
    return result.strip()

if __name__ == "__main__":
    cleaned = remove_inline_line_breaks(sample_source_code)
    print(f"Cleaned code:\n{cleaned}")
