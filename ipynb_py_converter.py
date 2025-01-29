import os
from nbconvert import PythonExporter
from nbformat import read, write

def convert_ipynb_to_py(ipynb_file, output_file=None):
    try:
        
        with open(ipynb_file, 'r', encoding='utf-8') as f:
            notebook = read(f, as_version=4)
        
        
        exporter = PythonExporter()
        source_code, _ = exporter.from_notebook_node(notebook)

        
        if output_file is None:
            output_file = os.path.splitext(ipynb_file)[0] + '.py'

        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(source_code)

        print(f"Converted {ipynb_file} to {output_file} successfully!")
    except Exception as e:
        print(f"Error: {e}")


convert_ipynb_to_py('../AIRL_Work/Scripts/Experiments/1.ipynb')  
