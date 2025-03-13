import os

def generate_file_structure(directory, indent=""): 
    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        print(indent + "[Permission Denied]")
        return
    
    for index, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        is_last = index == len(entries) - 1
        prefix = "└── " if is_last else "├── "
        print(indent + prefix + entry)
        
        if os.path.isdir(path):
            new_indent = indent + ("    " if is_last else "│   ")
            generate_file_structure(path, new_indent)
            
if __name__ == "__main__":
    folder_path ="../AIRL_Work/My Thesis"
    if os.path.isdir(folder_path):
        print(folder_path)
        generate_file_structure(folder_path)
    else:
        print("Invalid directory path.")
