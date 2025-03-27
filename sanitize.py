import os
import re
from pathlib import Path

# Define invalid characters for Windows
INVALID_CHARS = r'[<>:"/\\|?*]'

def sanitize_filename(filename):
    """Remove invalid characters and replace them with underscores."""
    return re.sub(INVALID_CHARS, '_', filename)

def sanitize_directory(root_dir):
    """Sanitize all files and directories in the given root directory."""
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for filename in filenames:
            old_path = Path(dirpath) / filename
            new_filename = sanitize_filename(filename)
            new_path = Path(dirpath) / new_filename
            if old_path != new_path:
                print(f"Renaming: {old_path} -> {new_path}")
                old_path.rename(new_path)

        for dirname in dirnames:
            old_path = Path(dirpath) / dirname
            new_dirname = sanitize_filename(dirname)
            new_path = Path(dirpath) / new_dirname
            if old_path != new_path:
                print(f"Renaming: {old_path} -> {new_path}")
                old_path.rename(new_path)

if __name__ == "__main__":
    repo_path = Path(__file__).parent  # Set this to your repository path
    sanitize_directory(repo_path)
    print("Sanitization complete. You can now commit safely.")