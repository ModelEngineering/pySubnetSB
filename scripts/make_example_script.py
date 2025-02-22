'''Makes the example notebook into a script.'''
import constants as cn  # type: ignore

import os
import nbformat # type: ignore
from nbconvert import PythonExporter # type: ignore

NOTEBOOK_NAME = "api_basics"
NOTEBOOK_PATH = os.path.join(cn.EXAMPLE_DIR, NOTEBOOK_NAME + ".ipynb")
SCRIPT_PATH = os.path.join(cn.EXAMPLE_DIR, NOTEBOOK_NAME + ".py")
ENDING_SCRIPT = """
if __name__ == '__main__':                                                  
    main()
    mp.freeze_support()
"""

# Load the notebook
with open(NOTEBOOK_PATH) as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert to Python script
python_exporter = PythonExporter()
python_script, _ = python_exporter.from_notebook_node(notebook_content)

# Process the script to make it executable
strings = python_script.split('\n')
found_is_plot = False
in_function = False
write_strings = []
indent = ""
for string in strings:
    write_strings.append(indent + string)
    # Parse until after the import statements
    if "IS_PLOT" in string:
        found_is_plot = True
    if not found_is_plot:
        continue
    if not in_function:
        # Turn off plotting
        write_strings.append("IS_PLOT = False")
        # Create the main funtion
        write_strings.append("def main():")
        in_function = True
        indent = "    "
# Verify result
if not found_is_plot:
    raise ValueError("IS_PLOT not found in the script.")
if not in_function:
    raise ValueError("Function not found in the script.")
# Add the script invocation calls
write_strings.append(ENDING_SCRIPT)
    
# Save the script
new_python_script = "\n".join(write_strings)
with open(SCRIPT_PATH, 'w') as f:
    f.write(new_python_script)