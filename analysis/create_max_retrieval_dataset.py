from datasets import Dataset, DatasetDict
import random
from tqdm import tqdm

def generate_data(num_samples=10000, min_list_length=0, max_list_length=8):
    data = []
    for _ in tqdm(range(num_samples)):

        upper_bound = random.randint(50, 1000)
        list_length = random.uniform(min_list_length, max_list_length)
        list_length = int(2**list_length + 0.5)
        integer_list = [random.randint(0, upper_bound) for _ in range(list_length)]
        max_value = max(integer_list)
        data.append({
            'problem': f'What is the maximum in the following sequence: {integer_list}?',
            'answer': max_value,
            'list_length': list_length
        })
    return data


random.seed(42)

# Generate the dataset
data = generate_data()

# Transform the list of dictionaries into a dictionary of lists
data_dict = {
    'problem': [item['problem'] for item in data],
    'answer': [item['answer'] for item in data],
    'list_length': [item['list_length'] for item in data]
}

# Create a Hugging Face Dataset
dataset = Dataset.from_dict(data_dict)

# Create a split for problems with list length < 2**4
easy_dataset = dataset.filter(lambda example: example['list_length'] < 2**5)

# Use the entire dataset without splitting and add the easy split
dataset_dict = DatasetDict({
    'full': dataset,
    'easy': easy_dataset
})

# Push the dataset to the Hugging Face Hub
dataset_dict.push_to_hub("mihirma/max-retrieval-dataset")