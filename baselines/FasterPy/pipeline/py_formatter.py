from typing import List, Set, Tuple
import tree_sitter_python as tspy
from tree_sitter import Language, Parser, Node, Tree, Query
from patch_synthesizer import syn
import re


# #####################################################################################################################
class Formatter:
    """
    Tree-sitter-based Python code formatting and cleanup utility.

    The main purpose of this class is to analyze the abstract syntax tree (AST)
    of Python code, trace function call chains, and remove redundant function
    definitions that are unused or do not need to be preserved.
    """

    def __init__(self):
        # Initialize the Tree-sitter Python language environment and parser.
        self.PY_LANGUAGE = Language(tspy.language())
        self.parser = Parser(self.PY_LANGUAGE)

    def __filter_functions(
        self,
        root_node: Tree,
        preserved_functions: Set[str],
        keep_function_signature: bool = False,
    ):
        """
        Filter and find all functions that do not need to be preserved.

        This method returns the line ranges of these functions, namely the code
        line hunks that need to be deleted.

        :param root_node: Root AST node of the source code.
        :param preserved_functions: Set of function names that need to be preserved.
        :param keep_function_signature: Whether to preserve function signatures.
                                        This parameter is not used in the current logic.
        :return: A list of code line ranges to delete, such as
                 [(start_row, end_row), ...]. Return None if no ranges exist.
        """
        # Use Tree-sitter query syntax to match all function definition nodes
        # and capture them as func_node.
        query_text = """
        (function_definition 
            name: (identifier)
        )@func_node
        """
        query = Query(self.PY_LANGUAGE, query_text)
        captures = query.captures(root_node)

        if not captures:
            return None

        function_nodes = captures["func_node"]
        delete_hunks = []

        # Iterate over all discovered function nodes.
        for function_node in function_nodes:
            # Extract the function name.
            # Decoding is required because Tree-sitter returns bytes.
            function_name = function_node.child_by_field_name("name").text.decode()

            # If the function is not in the preservation list, record its start
            # and end rows as a deletion range.
            if function_name not in preserved_functions:
                delete_hunks.append((function_node.start_point.row, function_node.end_point.row))

        if not delete_hunks:
            return None

        # Sort deletion ranges by their starting row.
        delete_hunks = sorted(delete_hunks, key=lambda x: x[0])

        def merge_hunks(input_hunks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            """
            Merge adjacent or overlapping deletion ranges to avoid conflicts
            when generating a patch.
            """
            merged_hunks = [input_hunks[0]]

            for start, end in input_hunks[1:]:
                last_start, last_end = merged_hunks[-1]
                if start <= last_end:
                    # Merge the current range with the previous range if they overlap.
                    merged_hunks[-1] = (last_start, max(last_end, end))
                else:
                    # Otherwise, add it as an independent range.
                    merged_hunks.append((start, end))

            return merged_hunks

        # Merge ranges and return the final deletion hunks.
        delete_hunks = merge_hunks(delete_hunks)
        return delete_hunks

    # Future improvement: match functions by both function name and parameters.
    def __find_function_nodes(self, root_node: Tree, function_names: Set[str]):
        """
        Find and return function definition nodes in the AST according to the
        specified set of function names.

        :param root_node: Root AST node.
        :param function_names: Set of function names to find.
        :return: A list of matched function nodes. Return None if no match is found.
        """
        if not function_names:
            return []

        # Dynamically build the query text and use the #any-of? predicate to
        # match multiple specified function names.
        query_text = f"""
        (function_definition 
            name: (identifier) @func_name
            (#any-of? @func_name {" ".join(function_names)})
        )@func_node
        """
        query = Query(self.PY_LANGUAGE, query_text)
        captures = query.captures(root_node)

        if not captures:
            return None

        function_nodes = captures["func_node"]
        return function_nodes

    def __get_callees(self, root_node: Tree, found_callees: Set[str]) -> Set[str]:
        """
        Recursively find all sub-functions, namely callees, invoked inside the
        target node, which is usually a function node.

        :param root_node: Current AST node being analyzed.
        :param found_callees: Set of already discovered callees, used for
                              deduplication and preventing infinite recursion.
        :return: Set of all callee function names.
        """
        # Query all function call syntax nodes under the target node.
        query_text = """
        (call
            function: (identifier) @callee_name_node
        )
        """
        query = Query(self.PY_LANGUAGE, query_text)
        captures = query.captures(root_node)

        if not captures:
            return set()

        # Extract all function names directly called inside the current node.
        callees = {name_node.text.decode() for name_node in captures["callee_name_node"]}

        for callee in callees:
            if callee in found_callees:
                continue

            found_callees.add(callee)

            # Note: using set(callee) would convert a string such as "func" into
            # {'f', 'u', 'n', 'c'}. Use {callee} or set([callee]) to avoid
            # potential matching errors.
            function_node = self.__find_function_nodes(root_node, set([callee]))
            if function_node:
                function_node = function_node[0]

                # Recursively find calls inside the callee function.
                sub_callees = self.__get_callees(function_node, found_callees)
                found_callees.update(sub_callees)

        return found_callees

    def __get_added_functions(self, old_root_node: Tree, new_root_node: Tree):
        """
        Compare the ASTs of the old code and new code to find newly added
        function names.

        :param old_root_node: Root AST node of the old code.
        :param new_root_node: Root AST node of the new code.
        :return: Set of newly added function names, which exist in the new node
                 but not in the old node.
        """
        query_text = """
        (function_definition 
            name: (identifier) @func_name
        )
        """
        query = Query(self.PY_LANGUAGE, query_text)

        # Extract all functions from the old code.
        old_captures = query.captures(old_root_node)
        if not old_captures:
            old_functions = []
        else:
            old_functions = [function_name.text.decode() for function_name in old_captures["func_name"]]

        # Extract all functions from the new code.
        new_captures = query.captures(new_root_node)
        if not new_captures:
            new_functions = []
        else:
            new_functions = [
                new_function_name.text.decode()
                for new_function_name in new_captures["func_name"]
            ]

        old_functions = set(old_functions)
        new_functions = set(new_functions)

        # Difference operation:
        # functions in the new code minus functions in the old code equals newly added functions.
        return new_functions - old_functions

    def __get_modified_functions(
        self,
        old_root_node: Tree,
        patch: str,
        context_length_offset: int = 3,
    ):
        """
        Find function names modified in the old code according to the content
        of a Git diff patch.

        :param old_root_node: Root AST node of the old code.
        :param patch: Diff string in unified diff format.
        :param context_length_offset: Context line offset in the diff, usually 3.
        :return: Set of modified function names.
        """
        # Use a regular expression to match hunk headers in the patch.
        # Example: @@ -10,5 +10,6 @@
        hunk_header_pattern = re.compile(r"\@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        matches = list(hunk_header_pattern.finditer(patch))
        hunks = []

        # Parse affected line ranges in the old file.
        for match in matches:
            old_start, old_count, _, _ = match.groups()
            old_count = old_count or "1"
            old_end = int(old_start) + int(old_count) - context_length_offset - 1
            old_start = int(old_start) + context_length_offset
            hunks.append((old_start, old_end))

        hunks = sorted(hunks, key=lambda x: x[0])
        modified_functions = set()

        # Extract all function nodes from the old code.
        query_text = """
        (function_definition 
            name: (identifier)
        )@func_node
        """
        query = Query(self.PY_LANGUAGE, query_text)
        captures = query.captures(old_root_node)

        if not captures:
            return set()

        function_nodes = captures["func_node"]

        for function_node in function_nodes:
            function_name = function_node.child_by_field_name("name").text.decode()

            # Tree-sitter row numbers start from 0, so convert them to 1-based line numbers.
            function_start_row = function_node.start_point.row + 1
            function_end_row = function_node.end_point.row + 1

            # Determine whether the current function line range intersects with
            # any modified range in the patch.
            for hunk_start, hunk_end in hunks:
                if not (hunk_start > function_end_row or hunk_end < function_start_row):
                    modified_functions.add(function_name)
                    break

        return modified_functions

    def __delete_useless_code(
        self,
        origin_code: str,
        root_node: Tree,
        preserved_functions: Set[str],
    ) -> str:
        """
        Perform the actual deletion operation.

        Based on the list of functions to preserve and their call relationships,
        this method generates and applies a patch to clean useless code.

        :param origin_code: Original code string.
        :param root_node: Root AST node of the original code.
        :param preserved_functions: Initial set of target functions to preserve.
        :return: Code string after useless code has been removed.
        """
        if not preserved_functions:
            return ""

        # Find all initially required function nodes to preserve.
        preserved_function_nodes = self.__find_function_nodes(root_node, preserved_functions) or []

        # Iterate over these base functions, recursively find all of their
        # dependencies, namely called sub-functions, and add them to the
        # preservation list.
        for preserved_function_node in preserved_function_nodes:
            callees = self.__get_callees(preserved_function_node, set())
            preserved_functions.update(callees)

        # Get all line ranges that are not in the preservation list and need to be deleted.
        delete_hunks = self.__filter_functions(root_node, preserved_functions)
        if not delete_hunks:
            return origin_code

        origin_lines = origin_code.splitlines()

        # Fabricate a unified diff patch header.
        patch = "--- example.txt\n+++ example.txt"
        deleted_line_count = 0

        # Construct patch hunks for deleting redundant functions.
        for start_row, end_row in delete_hunks:
            affected_rows = end_row - start_row + 1

            # Append the diff chunk header.
            patch += f"\n@@ -{start_row + 1},{affected_rows} +{start_row + 1 - deleted_line_count},0 @@"
            deleted_line_count += affected_rows

            # Append the specific code lines to be deleted with a leading minus sign.
            for i in range(start_row, end_row + 1):
                patch += f"\n-{origin_lines[i]}"

        # Key fix: manually append a newline at the end of the patch string to
        # satisfy the Linux patch format requirement.
        patch += "\n"

        # Use the patch_synthesizer library to apply the constructed patch to
        # the original code.
        patched_code = syn(origin_code, patch)

        if not patched_code:
            return origin_code

        return patched_code

    def format(self, old_code: str, target_function_name: str = "solution"):
        """
        Public interface.

        Generate modified file content according to the source file and
        target_function_name. This method deletes functions in the source code
        that are not modified and are not in the preservation list, where the
        preservation list includes the target function and its dependencies.
        It returns the cleaned source code string.
        """
        # Parse the source code and obtain its AST tree and root node.
        old_tree = self.parser.parse(old_code.encode("utf-8"))
        old_root_node = old_tree.root_node

        # Set the core preservation list: the entry function.
        preserved_functions = {target_function_name}

        # Execute the cleanup workflow.
        cleaned_origin_code = self.__delete_useless_code(
            old_code,
            old_root_node,
            preserved_functions,
        )
        return cleaned_origin_code


# #####################################################################################################################
if __name__ == "__main__":
    formatter = Formatter()

#   Demonstration test code. Commented out.
#   with open("example.py", "r") as file:
#       code = file.read()
#   with open("example-diff.patch", "r") as file:
#       patch = file.read()
#   cleaned_origin_code, cleaned_new_code = formatter.format(code, patch)
#   print(cleaned_new_code)

#   Demonstration of automated cleanup for competitive programming code:
#   cleaned_competitive_programming_code = formatter.format_cp(code)
#   print(cleaned_competitive_programming_code)