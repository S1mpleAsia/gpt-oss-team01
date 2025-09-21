# The name of your large file
source_file = '/nfs/gpu_trainee/getp08/final-project/gpt-oss-team01/evaluation/references/output_120b_token_ids_old.txt' 

# The name of the new file you want to create
destination_file = '/nfs/gpu_trainee/getp08/final-project/gpt-oss-team01/evaluation/references/output_120b_token_ids.txt' 

# The number of lines to copy
num_lines_to_copy = 1536

try:
    with open(source_file, 'r') as infile:
        # Read all lines from the source file
        lines = infile.readlines()

    with open(destination_file, 'w') as outfile:
        # Write the first 1000 lines to the new file
        # The slice [0:num_lines_to_copy] gets lines from index 0 up to (but not including) 1000
        outfile.writelines(lines[0:num_lines_to_copy])

    print(f"Successfully created '{destination_file}' with the first {num_lines_to_copy} lines from '{source_file}'.")

except FileNotFoundError:
    print(f"Error: The file '{source_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")