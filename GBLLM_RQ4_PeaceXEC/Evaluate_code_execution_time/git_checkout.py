import subprocess
import os


# #####################################################################################################################🔖💡🟨✓✗
DEBUG = False

# #####################################################################################################################🔖💡🟨✓✗
# ================== Git Repository Management ==================
def mark_repo_as_safe(repo_path):
    """
    Mark the repository as a safe Git directory.

    Args:
        repo_path (str): The absolute path to the repository.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", repo_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if DEBUG:
            print(f"\n✓✓✓ Repository '{repo_path}' has been marked as a safe directory.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗✗✗ Failed to mark '{repo_path}' as safe: {e.stderr.decode().strip()}")
        return False


# #####################################################################################################################🔖💡✅🟨❌
def switch_repo_to_version(repo_root_dir, version_id):
    """
    Switch the Git repository to the specified SHA commit.

    Args:
        repo_root_dir (str): The absolute path to the repository.
        version_id (str): The commit SHA to switch to.

    Returns:
        bool: True if the switch is successful, False otherwise.
    """
    if not os.getcwd().endswith(repo_root_dir):
        if not os.path.exists(repo_root_dir):
            print(f"✗✗✗ Repository path does not exist: {repo_root_dir}")
            return False

    # Mark the repository as safe
    if not mark_repo_as_safe(repo_root_dir):
        return False

    try:
        # Switch to the repository directory
        if not os.getcwd().endswith(repo_root_dir):
            os.chdir(repo_root_dir)

        # Checkout the specific commit
        subprocess.run(
            ["git", "checkout", version_id, "-f"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if DEBUG:
            print(f"✅✅✅ Repository switched to SHA '{version_id}', path is '{repo_root_dir}'.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌❌❌ Failed to switch repository '{repo_root_dir}' to SHA '{version_id}': {e.stderr.decode().strip()}")
        return False

    except Exception as e:
        print(f"❌❌❌ Unexpected error occurred while switching repository '{repo_root_dir}': {str(e)}")
        return False




# #####################################################################################################################🔖💡✅🟨❌
"""
Summary:

The goals of this code are:

    Mark a Git repository as a safe directory to avoid Git execution issues in untrusted directories.

    Switch the repository to a specified Git commit SHA, i.e., switch to a specific version of the code.

    
Applicable Scenarios:

    Used for scripted management of Git repositories, automatically switching to specified commit versions.

    Suitable for automated builds, CI/CD pipelines, code reviews, and other scenarios involving Git repository version switching and security configuration.

"""
# ================== Main Program Execution ==================
if __name__ == "__main__":
    # Example usage (replace with actual repository path and SHA)
    repo_path = "path/to/repository"  # Repository path
    commit_sha = "abcdef1234567890abcdef1234567890abcdef12"  # Example commit SHA

    switch_repo_to_version(repo_path, commit_sha)