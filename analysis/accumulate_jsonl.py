import argparse
import json
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Concatenate multiple JSONL files into one.")
    parser.add_argument('input_paths', nargs='+', help='Paths to the input JSONL files.')
    parser.add_argument('--output_path', required=True, help='Path to the output JSONL file.')
    return parser.parse_args()

def concatenate_jsonl_files(input_paths, output_path):
    with open(output_path, 'w') as outfile:
        for file_path in tqdm(input_paths):
            with open(file_path, 'r') as infile:
                for line in infile:
                    outfile.write(line)

def main():
    args = parse_arguments()
    concatenate_jsonl_files(args.input_paths, args.output_path)
    print(f"Concatenated files saved to {args.output_path}")

if __name__ == "__main__":
    main()
