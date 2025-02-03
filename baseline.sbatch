#!/bin/bash
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=15
#SBATCH --mem=50G
#SBATCH --constraint=a40
#SBATCH --job-name=run-charlie-baseline
#SBATCH --time=5:00:00
#SBATCH --output=/usr/stud/mahajanm/Projects/search-and-learn/logs/baseline-slurm-%j.out
#SBATCH --error=/usr/stud/mahajanm/Projects/search-and-learn/logs/baseline-slurm-%j.out


# Set the configuration file
export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml

# Run the Python script with the configuration
python scripts/test_time_compute.py $CONFIG