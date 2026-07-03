import os
import pandas as pd
from tqdm import tqdm
import ast
import astor
import io
import tokenize
import keyword

# Keywords indicating where inline line breaks should be preserved with a space
INLINE_BREAK_KEYWORDS = [
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
    'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally',
    'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal',
    'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'int', 'float'
]

DEBUG = False

# Example source code to clean
sample_source_code = """
"""

def main():
    simple_cleaned = simple_code_clean(sample_source_code)
    multi_cleaned = iterative_code_clean(sample_source_code)
    print(f"### Multi-pass cleaned code:\n{multi_cleaned}")

def simple_code_clean(code: str) -> str:
    """Perform a single-pass clean of the given source code."""
    cleaned = code
    cleaned = resolve_ast_replacements(cleaned)
    cleaned = ast_clean(cleaned)
    cleaned = remove_multiline_comments(cleaned)
    cleaned = remove_recursion_limits(cleaned.strip())
    cleaned = remove_main_guard(cleaned)
    cleaned = replace_is_operators(cleaned)
    cleaned = benign_replacements(cleaned)
    cleaned = remove_inline_line_breaks(cleaned)
    cleaned = remove_blank_lines(cleaned)
    return cleaned

def iterative_code_clean(code: str) -> str:
    """Perform multiple cleaning passes, including removal of logs, IO, exec, etc."""
    cleaned = code
    cleaned = resolve_ast_replacements(cleaned)
    cleaned = ast_clean(cleaned)
    cleaned = remove_multiline_comments(cleaned)
    cleaned = remove_recursion_limits(cleaned.strip())
    cleaned = remove_main_guard(cleaned)
    cleaned = replace_is_operators(cleaned)
    cleaned = benign_replacements(cleaned)

    cleaned = replace_strange_words(cleaned)
    cleaned = remove_logging(cleaned)
    cleaned = remove_process_creation(cleaned)
    cleaned = replace_strange_io(cleaned)
    cleaned = replace_exit_statements(cleaned)
    cleaned = remove_exec_calls(cleaned)

    cleaned = remove_inline_line_breaks(cleaned)
    cleaned = remove_blank_lines(cleaned)
    return cleaned

def resolve_ast_replacements(code_str: str) -> str:
    """Make simple string-based fixes before AST processing."""
    code_str = code_str.replace("print((*", "print(tuple(")
    code_str = code_str.replace("ÈëÁ¦Àý", "")
    code_str = code_str.replace("¤»¤¤¤¬¤ï", "fun")
    code_str = code_str.replace("¤Õ¤¬¤ï", "func")
    code_str = code_str.replace(
        'if os.name == "posix":\n    ', 
        'if os.name == "posix":\n    pass\n    '
    )
    return code_str.strip()

def ast_clean(code_str: str) -> str:
    """Parse and unparse the code via AST to normalize formatting."""
    try:
        tree = ast.parse(code_str.strip())
        cleaned = astor.to_source(tree)
        return cleaned.strip()
    except Exception:
        print(f"### AST parse failed for code:\n{code_str}")
        return code_str

class CommentRemover(ast.NodeTransformer):
    """AST transformer to remove standalone string comments."""
    def visit(self, node):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
            return None
        return super().visit(node)

def remove_multiline_comments(code_str: str) -> str:
    """Strip out multiline string comments via AST."""
    try:
        tree = ast.parse(code_str.strip())
        remover = CommentRemover()
        new_tree = remover.visit(tree)
        cleaned = astor.to_source(new_tree).strip()
        return cleaned
    except Exception:
        return code_str

def remove_recursion_limits(code: str) -> str:
    """Remove setrecursionlimit or stack_size calls."""
    if "setrecursionlimit(" not in code and "stack_size(" not in code:
        return code
    lines = code.split("\n")
    kept = []
    for line in lines:
        if "setrecursionlimit(" in line or "stack_size(" in line:
            if DEBUG:
                print(f"### Dropping recursion limit call: {line}")
        else:
            kept.append(line)
    return "\n".join(kept).strip()

def remove_main_guard(code: str) -> str:
    """Unwrap code under if __name__ == '__main__' guard."""
    if "__name__" not in code or "__main__" not in code:
        return code
    try:
        tree = ast.parse(code.strip())
        for idx, node in enumerate(tree.body):
            if isinstance(node, ast.If):
                test = node.test
                # Detect: if __name__ == "__main__"
                if (isinstance(test.left, ast.Name) and test.left.id == "__name__" and
                    isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == "__main__"):
                    tree.body[idx:idx+1] = node.body
        cleaned = astor.to_source(tree).strip()
        return cleaned
    except Exception:
        return code

def replace_is_operators(code: str) -> str:
    """Convert 'is' and 'is not' to '==' and '!='."""
    if " is not " not in code and " is " not in code:
        return code
    code = code.replace(" is not ", " != ")
    code = code.replace(" is ", " == ")
    return code.strip()

def benign_replacements(code_str: str) -> str:
    """Replace standard I/O calls with input()/print()."""
    for pattern in ["sys.stdin.readline", "stdin.readline", "sys.stdin.buffer.readline", "stdin.buffer.readline"]:
        if pattern in code_str:
            code_str = code_str.replace(pattern, "input")
    for pattern in [
        "sys.stdout.write", "stdout.write", "sys.__stdout__.write", "__stdout__.write",
        "sys.stderr.write", "stderr.write", "sys.__stderr__.write", "__stderr__.write"
    ]:
        if pattern in code_str:
            code_str = code_str.replace(pattern, "print")
    if 'return sys.stdout.flush()' in code_str:
        code_str = code_str.replace('return sys.stdout.flush()', "return")
    if 'return stdout.flush()' in code_str:
        code_str = code_str.replace('return stdout.flush()', "return")
    return code_str.strip()

def replace_strange_words(code_str: str) -> str:
    """Clean up unusual import and I/O constructs."""
    def fix_imports(s: str) -> str:
        lines = s.split("\n")
        out = []
        for line in lines:
            if "import" in line and any(x in line for x in ["stdout", "stderr"]):
                if DEBUG:
                    print(f"### Before import fix: {line}")
                line = line.replace("from sys import stdout, ", "from sys import ")
                line = line.replace(", stdout", "")
                line = line.replace("from sys import stdout", "")
                line = line.replace("print = __import__('sys').stdout.write", "")
                line = line.replace("from sys import stderr, ", "from sys import ")
                line = line.replace(", stderr", "")
                line = line.replace("from sys import stderr", "")
                line = line.replace("print = __import__('sys').stderr.write", "")
                if line.strip():
                    out.append(line)
                if DEBUG and any(x in line for x in ["stdout", "stderr"]):
                    print(f"### After import fix: {line}")
            else:
                out.append(line)
        return "\n".join(out).strip()

    if "import" in code_str and any(x in code_str for x in ["stdout", "stderr"]):
        code_str = fix_imports(code_str)

    for pattern in [", file=sys.stdout", ", file=stdout", ", output=sys.stdout", ", output=stdout",
                    ", file=sys.stderr", ", file=stderr", ", output=sys.stderr", ", output=stderr"]:
        if pattern in code_str:
            code_str = code_str.replace(pattern, "")

    def fix_strange_io(s: str) -> str:
        lines = s.split("\n")
        out = []
        for line in lines:
            if ('class ' not in line and 
                any(x in line for x in ["IOWrapper(", "FastIO(", "StringIO(", "BytesIO(", "FastStdout(", ".close(", "stdout", "stderr"])):
                indent = line[:len(line) - len(line.lstrip())]
                out.append(f"{indent}pass")
            else:
                out.append(line)
        return "\n".join(out).strip()

    if any(x in code_str for x in ["IOWrapper(", "FastIO(", "StringIO(", "BytesIO(", "FastStdout(", ".close(", "stdout", "stderr"]):
        code_str = fix_strange_io(code_str)

    def fix_open_calls(s: str) -> str:
        lines = s.split("\n")
        out = []
        for line in lines:
            if 'open(0' in line:
                out.append(line)
            elif 'open(' in line:
                indent = line[:len(line) - len(line.lstrip())]
                out.append(f"{indent}pass")
            else:
                out.append(line)
        return "\n".join(out).strip()

    if 'open(' in code_str:
        code_str = fix_open_calls(code_str)

    return code_str.strip()

def remove_logging(code_str: str) -> str:
    """Replace logging and system calls with pass."""
    patterns = ["unittest.", "traceback.", "os.write", "logging.", "os.system", "import division", "__future__"]
    if not any(x in code_str for x in patterns):
        return code_str
    lines = code_str.split("\n")
    out = []
    for line in lines:
        if 'class ' not in line and any(x in line for x in patterns):
            indent = line[:len(line) - len(line.lstrip())]
            out.append(f"{indent}pass")
            if DEBUG:
                print(f"### Removing logging line: {line}")
        else:
            out.append(line)
    return "\n".join(out).strip()

def remove_process_creation(code_str: str) -> str:
    """Strip out threading and multiprocessing constructs."""
    def fix_imports(s: str) -> str:
        lines = s.split("\n")
        out = []
        for line in lines:
            if "import" in line and "threading" in line:
                line = line.replace("import threading, ", "import ")
                line = line.replace(", threading", "")
                line = line.replace("import threading", "")
                if line.strip():
                    out.append(line)
            else:
                out.append(line)
        return "\n".join(out).strip()

    if "import" in code_str and "threading" in code_str:
        code_str = fix_imports(code_str)

    def fix_thread_calls(s: str) -> str:
        lines = s.split("\n")
        out = []
        for line in lines:
            if "threading.Thread(target=" in line:
                fn = line.split("threading.Thread(target=")[1].split(")")[0]
                out.append(f"{fn}()")
            else:
                out.append(line)
        return "\n".join(out).strip()

    if 'threading.Thread(target=' in code_str:
        code_str = fix_thread_calls(code_str)

    def drop_start_join(s: str) -> str:
        lines = s.split("\n")
        out = []
        for line in lines:
            if ".start()" in line or ".join()" in line:
                indent = line[:len(line) - len(line.lstrip())]
                out.append(f"{indent}pass")
            else:
                out.append(line)
        return "\n".join(out).strip()

    if ".start()" in code_str or ".join()" in code_str:
        code_str = drop_start_join(code_str)

    def drop_remaining(s: str) -> str:
        patterns = [
            "threading", "thread", "multiprocessing", "asyncio", "queue.Queue(", 
            "ProcessPoolExecutor", "concurrent", "fork(", "subprocess.run("
        ]
        lines = s.split("\n")
        out = []
        for line in lines:
            if any(x in line for x in patterns):
                indent = line[:len(line) - len(line.lstrip())]
                out.append(f"{indent}pass")
                if DEBUG:
                    print(f"### Dropping process line: {line}")
            else:
                out.append(line)
        return "\n".join(out).strip()

    patterns = ["threading", "thread", "multiprocessing", "asyncio", "queue.Queue(", 
                "ProcessPoolExecutor", "concurrent", "fork(", "subprocess.run("]
    if any(x in code_str for x in patterns):
        code_str = drop_remaining(code_str)

    return code_str.strip()

def replace_strange_io(code_str: str) -> str:
    """Replace I/O classes and buffer constructs with pass."""
    patterns = ["IOWrapper", "FastIO", "StringIO", "BytesIO", "FastStdout", "IOBase", "setrecursionlimit"]
    if not any(x in code_str for x in patterns):
        return code_str
    lines = code_str.split("\n")
    out = []
    for line in lines:
        if any(x in line for x in patterns):
            indent = line[:len(line) - len(line.lstrip())]
            out.append(f"{indent}pass")
            if DEBUG:
                print(f"### Replacing IO line: {line}")
        else:
            out.append(line)
    return "\n".join(out).strip()

def replace_exit_statements(code_str: str) -> str:
    """Convert exit() or quit() calls to return statements."""
    def fix_exits(s: str) -> str:
        lines = s.split("\n")
        out = []
        for line in lines:
            if 'exit(' in line or 'quit(' in line:
                stripped = line.lstrip()
                indent = line[:len(line) - len(stripped)]
                out.append(f"{indent}return")
            else:
                out.append(line)
        return "\n".join(out).strip()
    if 'exit(' in code_str or 'quit(' in code_str:
        code_str = fix_exits(code_str)
    return code_str.strip()

def remove_exec_calls(code_str: str) -> str:
    """Remove exec() calls by replacing with pass."""
    if "exec(" not in code_str:
        return code_str
    lines = code_str.split("\n")
    out = []
    for line in lines:
        if "exec(" in line:
            indent = line[:len(line) - len(line.lstrip())]
            out.append(f"{indent}pass")
        else:
            out.append(line)
    return "\n".join(out).strip()

def remove_inline_line_breaks(source: str) -> str:
    """Merge lines broken by tokenize.NL tokens, preserving keywords."""
    lines = source.strip().split('\n')
    tokens = tokenize.generate_tokens(io.StringIO(source.strip()).readline)
    drop_indices = []
    for tok_type, tok_str, start, end, _ in tokens:
        if tok_type == tokenize.NL and start[1] > 5:
            drop_indices.append(start[0] - 1)
    if not drop_indices:
        return source.strip()
    drop_indices.reverse()
    for idx in drop_indices:
        head = lines[idx].rstrip()
        tail = lines[idx + 1].lstrip()
        space_needed = any(
            head.endswith(k) or tail.startswith(k) for k in INLINE_BREAK_KEYWORDS
        )
        if space_needed:
            lines[idx] = f"{head} {tail}"
        else:
            lines[idx] = f"{head}{tail}"
        del lines[idx + 1]
    return '\n'.join(lines).strip()

def remove_blank_lines(code_str: str) -> str:
    """Strip out empty lines."""
    return "\n".join(line for line in code_str.split("\n") if line.strip()).strip()

if __name__ == "__main__":
    main()
