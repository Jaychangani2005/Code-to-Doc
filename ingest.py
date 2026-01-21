import os
import shutil
import git  # pip install GitPython

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

# Directories to ignore while scanning
IGNORED_DIRS = {
    '.git', '.idea', '__pycache__', 'venv', 'env',
    'node_modules', 'dist', 'build', '.vscode'
}

# Allowed source code file extensions
ALLOWED_EXTENSIONS = {
    '.py', '.java', '.js', '.ts', '.cpp', '.c', '.h', '.cs', '.go', '.rb'
}

# Default local directory to clone repo
LOCAL_REPO_DIR = "./repo_data"

# -------------------------------------------------
# STEP 1: CLONE REPOSITORY
# -------------------------------------------------

def clone_repository(repo_url, local_dir=LOCAL_REPO_DIR):
    """
    Clones a GitHub repository to a local directory.
    """

    # Clean existing directory (important for re-runs)
    if os.path.exists(local_dir):
        print(f"🧹 Cleaning up existing data in {local_dir}...")

        def on_rm_error(func, path, exc_info):
            os.chmod(path, 0o777)
            func(path)

        try:
            shutil.rmtree(local_dir, onerror=on_rm_error)
        except Exception as e:
            print(f"❌ Error cleaning directory: {e}")
            return None

    # Clone repo
    try:
        print(f"⬇️ Cloning repository...")
        git.Repo.clone_from(repo_url, local_dir)
        print(f"✅ Repository cloned successfully to: {local_dir}")
        return local_dir

    except git.exc.GitCommandError as e:
        print("❌ Failed to clone repository.")
        print(f"Details: {e}")
        return None

# -------------------------------------------------
# STEP 2: SCAN CODE FILES
# -------------------------------------------------

def get_code_files(root_dir):
    """
    Scans the directory for valid code files, ignoring junk folders.
    """

    code_files = []
    print(f"\n🔍 Scanning {root_dir} for code files...")

    for root, dirs, files in os.walk(root_dir):
        # Prevent walking into ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for file in files:
            _, ext = os.path.splitext(file)
            if ext in ALLOWED_EXTENSIONS:
                full_path = os.path.join(root, file)
                code_files.append(full_path)

    print(f"✅ Found {len(code_files)} valid code files.")
    return code_files

# -------------------------------------------------
# MAIN EXECUTION FLOW
# -------------------------------------------------

if __name__ == "__main__":
    repo_url = input("Enter the GitHub Repository URL: ").strip()

    cloned_path = clone_repository(repo_url)

    if cloned_path:
        code_files = get_code_files(cloned_path)

        print("\n--- Sample Code Files ---")
        for file in code_files[:5]:
            print(file)

