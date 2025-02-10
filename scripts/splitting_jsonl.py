import json
import os
import argparse

def split_jsonl(file_path, num_splits=4):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)
    chunk_size = total_lines // num_splits
    remainder = total_lines % num_splits

    base_name, ext = os.path.splitext(file_path)
    
    start = 0
    for i in range(num_splits):
        end = start + chunk_size + (1 if i < remainder else 0)
        split_file = f"{base_name}_part_{i}{ext}"
        
        with open(split_file, "w", encoding="utf-8") as f_out:
            f_out.writelines(lines[start:end])
        
        print(f"Created {split_file} with {end-start} lines")
        start = end

def main():
    parser = argparse.ArgumentParser(description="Split a JSONL file into multiple parts.")
    parser.add_argument("--filepath", required=True, help="Path to the JSONL file to be split")
    args = parser.parse_args()

    split_jsonl(args.filepath, num_splits=8)

if __name__ == "__main__":
    main()
