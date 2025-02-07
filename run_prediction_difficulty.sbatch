#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=hai_hreplearn
#SBATCH --cpus-per-task=12
#SBATCH --output=logs_dif/slurm_%j.log
#SBATCH --error=logs_dif/slurm_%j.log
#SBATCH --mail-user=avocadoaling@gmail.com
#SBATCH --mail-type=ALL

# load the environment

source sc_venv_template/activate.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "starting ttt at time $(date +%Y-%m-%d_%H-%M-%S)"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"


CUDA_VISIBLE_DEVICES=0 python scripts/test_time_compute.py analysis/adaptive_best_of_n_toy_example.yaml --dataset_name='data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions_analysis/difficulty.msgpack' &
CUDA_VISIBLE_DEVICES=1 python scripts/test_time_compute.py analysis/adaptive_best_of_n_toy_example.yaml --dataset_name='data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions_analysis/difficulty_clipped.msgpack' &
CUDA_VISIBLE_DEVICES=2 python scripts/test_time_compute.py analysis/adaptive_best_of_n_toy_example.yaml --dataset_name='data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions_analysis/difficulty_mean.msgpack' &
CUDA_VISIBLE_DEVICES=3 python scripts/test_time_compute.py analysis/adaptive_best_of_n_toy_example.yaml --dataset_name='data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions_analysis/difficulty_mean_clipped.msgpack' &

# Wait for all background processes to complete
wait

echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"

echo "Done at $(date +%Y-%m-%d_%H-%M-%S)"

echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"