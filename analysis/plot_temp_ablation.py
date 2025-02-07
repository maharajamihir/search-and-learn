import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_combined_pass_at_n(json_file_path: Path):
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract temperatures and pass_at_* values
    temperatures = []
    pass_at_n_values = []
    pass_at_n_clipped_values = []
    pass_at_n_constant_values = []

    for key, value in data.items():
        # Extract temperature from the key
        temp_str = key.split('_')[-2]
        temperature = float(temp_str)
        temperatures.append(temperature)

        # Extract pass_at_* values
        pass_at_n_values.append(value["64"]["pass_at_n"])
        pass_at_n_clipped_values.append(value["64"]["pass_at_n_clipped"])
        pass_at_n_constant_values.append(value["64"]["pass_at_n_constant"])

    # Sort data by temperature
    sorted_indices = sorted(range(len(temperatures)), key=lambda i: temperatures[i])
    temperatures = [temperatures[i] for i in sorted_indices]
    pass_at_n_values = [pass_at_n_values[i] for i in sorted_indices]
    pass_at_n_clipped_values = [pass_at_n_clipped_values[i] for i in sorted_indices]
    pass_at_n_constant_values = [pass_at_n_constant_values[i] for i in sorted_indices]

    # Plot all pass_at_* values in one plot
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, pass_at_n_values, marker='o', label='Pass@N')
    plt.plot(temperatures, pass_at_n_clipped_values, marker='o', label='Pass@N Clipped')
    plt.plot(temperatures, pass_at_n_constant_values, marker='o', label='Pass@N Constant')
    plt.title('Pass@N Metrics vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Pass@N Metrics')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(json_file_path.parent / 'combined_pass_at_n_vs_temperature_64.png')
    plt.close()

# Specify the path to your JSON file
json_file_path = Path('data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions_temperature_ablation/pass_at_n_temp_ablation_64.json')
print(json_file_path)
plot_combined_pass_at_n(json_file_path)