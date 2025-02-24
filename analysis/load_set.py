import argparse
from datasets import load_dataset

def load_and_save_huggingface_dataset(dataset_name, split, output_file_path):
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)
    
    # Save the dataset to a JSONL file
    dataset.to_json(output_file_path, lines=True)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load a HuggingFace dataset and save it to a JSONL file.")
    parser.add_argument("dataset_name", type=str, help="The name of the dataset to load (e.g., 'imdb').")
    parser.add_argument("split", type=str, help="The split of the dataset to load (e.g., 'train').")
    parser.add_argument("output_file_path", type=str, help="The path to save the output JSONL file (e.g., 'output.jsonl').")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load and save the dataset
    load_and_save_huggingface_dataset(args.dataset_name, args.split, args.output_file_path)

if __name__ == "__main__":
    main()


