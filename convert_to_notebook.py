import json
import re

def python_to_notebook(python_file, notebook_file):
    """
    Convert a Python file with triple-quoted strings as cell content
    into a Jupyter notebook.
    """
    with open(python_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract cells using regex
    # This pattern matches triple-quoted strings (''' or """)
    pattern = r"'''(.*?)'''"
    cells_content = re.findall(pattern, content, re.DOTALL)
    
    # Create notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Process each cell
    for cell_content in cells_content:
        # Determine cell type (markdown or code)
        if cell_content.lstrip().startswith('#'):
            # Markdown cell
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": cell_content.split('\n')
            })
        else:
            # Code cell
            notebook["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell_content.split('\n')
            })
    
    # Write the notebook to a file
    with open(notebook_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Notebook created: {notebook_file}")

if __name__ == "__main__":
    python_to_notebook("assignment5_notebook_content.py", "Assignment5_Solution.ipynb") 