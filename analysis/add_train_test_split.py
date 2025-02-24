from tqdm import tqdm
from datasets import load_dataset, Dataset
import msgpack

file_path = "data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions-15_analysis/analysis.msgpack"

with open(file_path, 'rb') as f:
    results = msgpack.unpackb(f.read())



dataset = load_dataset("mihirma/max-retrieval-dataset")

# Convert the list of results into a dictionary of lists using tqdm
results_dict = {
    'problem': [item['problem'] for item in tqdm(results, desc="Processing problems")],
    'answer': [item['answer'] for item in tqdm(results, desc="Processing answers")],
    'list_length': [item['list_length'] for item in tqdm(results, desc="Processing list lengths")]
}

# Create a Hugging Face Dataset from the results
test_dataset = Dataset.from_dict(results_dict)

# Add the test dataset to the existing dataset as a new split
dataset['test'] = test_dataset

# Subtract the "test" split from the "full" split to create the "train" split
full_dataset = dataset['full']
test_problems = set(tqdm(test_dataset['problem'], desc="Collecting test problems"))

# Filter out the test problems from the full dataset to create the train dataset
train_dataset = full_dataset.filter(lambda example: example['problem'] not in test_problems)

# Add the train dataset to the existing dataset as a new split
dataset['train'] = train_dataset

dataset.push_to_hub("mihirma/max-retrieval-dataset")
