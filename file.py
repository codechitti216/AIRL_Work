import os

def generate_file_structure(path, indent=0):
    # Check if the given path exists
    if not os.path.exists(path):
        print(f"Path '{path}' does not exist.")
        return

    # Get all files and directories in the path
    files_and_dirs = os.listdir(path)

    for file_or_dir in files_and_dirs:
        full_path = os.path.join(path, file_or_dir)

        # Print the current file/folder with indentation
        print(' ' * indent + file_or_dir)

        # If it's a directory, recursively list its contents
        if os.path.isdir(full_path):
            generate_file_structure(full_path, indent + 4)  # Increase indentation for subdirectories

# Example usage
folder_path = "../AIRL_Work"  # Replace with your folder path
generate_file_structure(folder_path)
