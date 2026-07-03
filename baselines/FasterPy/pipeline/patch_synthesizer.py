# This file is mainly responsible for synthesizing the generated patch with
# the original code, thereby producing the final optimized or modified faster code.
import subprocess
import os
import difflib
import tempfile

# Define the standard patch-format marker used when a file is missing
# a trailing newline at the end.
NO_NEWLINE_AT_EOF_MARKER = r"\ No newline at end of file"


def generate_unified_diff(old_code: str, new_code: str, context_length: int = 3):
    """
    Compare old and new code and generate a unified diff.

    :param old_code: Original code before modification.
    :param new_code: New code after modification.
    :param context_length: Number of context lines in the diff. The default is 3.
    :return: Patch text in unified diff format.
    """
    # Split the old and new code by lines.
    old_code_lines = old_code.splitlines()
    new_code_lines = new_code.splitlines()

    # Use difflib to generate a standard unified diff.
    patch_lines = difflib.unified_diff(
        old_code_lines,
        new_code_lines,
        "old_code",
        "new_code",
        "None",
        "None",
        n=context_length,
    )
    patch_lines = list(patch_lines)

    # Handle the edge case where the file does not end with a newline,
    # following the standard patch syntax.
    if not old_code.endswith("\n"):
        patch_lines.insert(-1, NO_NEWLINE_AT_EOF_MARKER)
    if not new_code.endswith("\n"):
        patch_lines.append(NO_NEWLINE_AT_EOF_MARKER)

    # Remove redundant trailing newline characters from each line and join them
    # into a complete patch string with single newline separators.
    patch = "\n".join([line.rstrip("\n") for line in patch_lines])
    return patch


def apply_patch(code: str, patch: str, reverse: bool = False) -> str:
    """
    Apply a patch to the original code using the system-level `patch` command.

    :param code: Original code.
    :param patch: Patch text to apply.
    :param reverse: Whether to apply the patch in reverse, namely undo the patch.
    :return: Complete merged code string. Return an empty string if the merge fails.
    """
    # If there is no patch content, return the original code directly.
    if not patch:
        return code

    has_retried = False

    while True:
        # Create a safe named temporary file and write the original code into it
        # so that the `patch` command can read and modify it.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            # Apply the patch normally.
            if not reverse:
                process = subprocess.run(
                    ["patch", temp_file_path],
                    input=patch,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

            # Apply the patch in reverse using the -R option.
            else:
                process = subprocess.run(
                    ["patch", "-R", temp_file_path],
                    input=patch,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=3,
                    text=True,
                )

            # A non-zero return code indicates that patch application failed.
            if process.returncode != 0:
                # Fault-tolerance mechanism for the first failure:
                # the mismatch may be caused by an extra trailing newline
                # in the original code. Remove the trailing newline and retry.
                if not has_retried:
                    has_retried = True
                    code = code.rstrip("\n")
                    continue

                # If the retry still fails, print error information and return an empty string.
                else:
                    print("Patch return code:", process.returncode)
                    print("Patch stdout:", process.stdout)
                    print("Patch stderr:", process.stderr)

                    print("Original code:\n", code)
                    print("Original patch:\n", patch)
                    return ""

            else:
                # The patch was applied successfully. Exit the retry loop.
                break

        # Catch subprocess timeout exceptions.
        except subprocess.TimeoutExpired:
            if not has_retried:
                has_retried = True
                code = code.rstrip("\n")
                continue
            else:
                os.remove(temp_file_path)
                return ""

        # Catch all other unknown exceptions.
        except Exception:
            os.remove(temp_file_path)
            return ""

    # Read the temporary file after the patch has been successfully applied.
    with open(temp_file_path, "r") as file:
        patched_code = file.read()

    # Remove the temporary file after use.
    os.remove(temp_file_path)

    return patched_code


def convert_git_diff_to_unified_diff(git_patch: str):
    """
    Convert a Git diff into the standard unified diff format.

    Git diff usually contains version-control metadata in the first two lines,
    such as 'diff --git ...' and 'index ...'. After removing the first two lines,
    the remaining content is the standard unified diff.
    """
    # Split the input string into lines.
    lines = git_patch.splitlines()

    # Skip the first two lines.
    unified_diff_lines = lines[2:]

    # Join the remaining lines back into a string.
    return "\n".join(unified_diff_lines)


# Local test module.
if __name__ == "__main__":
    # Read the original test file.
    original_test_code = ""
    with open("example.py", "r") as file:
        original_test_code = file.read()

    # Read the target test file.
    target_test_code = ""
    with open("example2.py", "r") as file:
        target_test_code = file.read()

    # 1. Compare the two files and generate a patch.
    patch_text = generate_unified_diff(original_test_code, target_test_code)

    # 2. If there is no difference, skip the patch process.
    # Otherwise, call apply_patch for synthesis testing.
    if not patch_text:
        print("No changes detected. Skipping the patch process.")
    else:
        synthesized_code = apply_patch(original_test_code, patch_text)
        print("Synthesis completed.")
        print(synthesized_code)