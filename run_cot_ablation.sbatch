#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=hai_hreplearn
#SBATCH --cpus-per-task=12
#SBATCH --output=logs_ttt/slurm_%j.log
#SBATCH --error=logs_ttt/slurm_%j.log
#SBATCH --mail-user=avocadoaling@gmail.com
#SBATCH --mail-type=ALL

# load the environment

source sc_venv_template/activate.sh
start_time=$(date +%s)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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


CUDA_VISIBLE_DEVICES=0 python scripts/test_time_compute.py analysis/adaptive-max-retreival.yaml --dataset_name=data/cot_ablation/best_of_n_completions_part_0_analysis/analysis.msgpack 2>&1 | tee logs/logs_cot/slurm_0_%j.log & 
CUDA_VISIBLE_DEVICES=1 python scripts/test_time_compute.py analysis/adaptive-max-retreival.yaml --dataset_name=data/cot_ablation/best_of_n_completions_part_1_analysis/analysis.msgpack 2>&1 | tee logs/logs_cot/slurm_1_%j.log &
CUDA_VISIBLE_DEVICES=2 python scripts/test_time_compute.py analysis/adaptive-max-retreival.yaml --dataset_name=data/cot_ablation/best_of_n_completions_part_2_analysis/analysis.msgpack 2>&1 | tee logs/logs_cot/slurm_2_%j.log &
CUDA_VISIBLE_DEVICES=3 python scripts/test_time_compute.py analysis/adaptive-max-retreival.yaml --dataset_name=data/cot_ablation/best_of_n_completions_part_3_analysis/analysis.msgpack 2>&1 | tee logs/logs_cot/slurm_3_%j.log &

# Wait for all background processes to complete
wait

echo "================================================"
echo "================================================"
echo "Total time taken: $(( $(date +%s) - start_time )) seconds"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"