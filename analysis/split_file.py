import json
import sys
from pathlib import Path

def split_jsonl_file(input_file_path, n):
    # Ensure the input file exists
    input_file = Path(input_file_path)
    if not input_file.exists():
        raise FileNotFoundError(f"The file {input_file_path} does not exist.")

    # Read the input file and count the number of lines
    with open(input_file, 'r') as f:
        lines = f.readlines()

    num_lines = len(lines)
    if n <= 0 or n > num_lines:
        raise ValueError("The number of splits must be between 1 and the number of lines in the file.")

    # Calculate the number of lines per split
    lines_per_split = num_lines // n
    remainder = num_lines % n

    # Split the lines into n parts
    start = 0
    for i in range(n):
        end = start + lines_per_split + (1 if i < remainder else 0)
        split_lines = lines[start:end]
        start = end

        # Write each split to a new file
        output_file_path = input_file.with_name(f"{input_file.stem}_part_{i+1}.jsonl")
        with open(output_file_path, 'w') as output_file:
            output_file.writelines(split_lines)
        print(f"Created {output_file_path} with {len(split_lines)} lines.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_file.py <path_to_jsonl> <number_of_splits>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    try:
        n = int(sys.argv[2])
    except ValueError:
        print("The number of splits must be an integer.")
        sys.exit(1)

    split_jsonl_file(input_file_path, n)
