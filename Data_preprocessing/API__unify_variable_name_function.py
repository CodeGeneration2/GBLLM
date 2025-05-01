import ast
import astor
from API__Remove_Inline_Breaks import remove_inline_line_breaks

# Sample code to be normalized
template_code = r"""
from itertools import combinations, count
n = int(eval(input()))
r = list(range(n))
a = [(set(), set()) for _ in r]
for i in r:
    for _ in range(int(eval(input()))):
        x, y = list(map(int, input().split()))
        a[i][y].add(x - 1)
r = next(i for i in count(n, -1)
         for x in map(set, combinations(r, i))
         if all(a[j][0].isdisjoint(x) and a[j][1] < x for j in x))
print(r)
"""

def normalize_variable_names(code: str) -> str:
    """
    Rename all function and variable identifiers to generic names (varN, funcN).
    Uses a three-pass approach with an initial temp naming, positional sorting,
    and a final rename, then removes inline line breaks for consistency.
    """
    # Parse code into AST
    try:
        tree = ast.parse(code)
    except Exception:
        return code

    var_count = 0
    func_count = 0
    renamed_vars = {}
    renamed_funcs = {}

    # First pass: assign temporary names to args, variables, and functions
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            original = node.arg
            if original not in renamed_vars:
                var_count += 1
                renamed_vars[original] = f"temp_var_{var_count}"
            node.arg = renamed_vars[original]
        if isinstance(node, ast.Name):
            original = node.id
            if original in renamed_vars:
                node.id = renamed_vars[original]
            elif isinstance(node.ctx, ast.Load):
                continue
            else:
                var_count += 1
                renamed_vars[original] = f"temp_var_{var_count}"
                node.id = renamed_vars[original]
        if isinstance(node, ast.FunctionDef):
            original = node.name
            func_count += 1
            renamed_funcs[original] = f"func_{func_count}"
            node.name = renamed_funcs[original]

    # Second pass: ensure all function names are updated in Name nodes
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in renamed_funcs:
            node.id = renamed_funcs[node.id]
        if isinstance(node, ast.arg) and node.arg in renamed_vars:
            node.arg = renamed_vars[node.arg]

    # Prepare for final renaming: extract generated code and locate temp names
    messy_code = astor.to_source(tree)
    positions = []
    pos_to_temp = {}
    for i in range(1, var_count + 1):
        temp_name = f"temp_var_{i}"
        idx = messy_code.find(temp_name)
        positions.append(idx)
        pos_to_temp[idx] = temp_name

    # Sort by position descending and map to final var names
    sorted_positions = sorted(positions, reverse=True)
    final_map = {}
    current = var_count
    for pos in sorted_positions:
        old = pos_to_temp[pos]
        final_map[old] = f"var{current}"
        current -= 1

    # Third pass: apply final renaming to the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.arg in final_map:
            node.arg = final_map[node.arg]
        if isinstance(node, ast.Name) and node.id in final_map:
            node.id = final_map[node.id]

    # Unparse and clean up inline breaks
    cleaned_code = astor.to_source(tree)
    cleaned_code = remove_inline_line_breaks(cleaned_code)
    return cleaned_code.strip()

if __name__ == '__main__':
    normalized = normalize_variable_names(template_code)
    print(normalized)
