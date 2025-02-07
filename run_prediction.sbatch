#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
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


CUDA_VISIBLE_DEVICES=0 python scripts/test_time_compute.py analysis/dispersion_toy_example.yaml  --temperature=0.0 --search_batch_size=1 > logs_ttt/slurm_0_%j.log 2>&1 & 
CUDA_VISIBLE_DEVICES=1 python scripts/test_time_compute.py analysis/dispersion_toy_example.yaml  --temperature=0.2 --search_batch_size=1 > logs_ttt/slurm_1_%j.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/test_time_compute.py analysis/dispersion_toy_example.yaml  --temperature=0.4 --search_batch_size=1 > logs_ttt/slurm_2_%j.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/test_time_compute.py analysis/dispersion_toy_example.yaml  --temperature=0.6 --search_batch_size=1 > logs_ttt/slurm_3_%j.log 2>&1 &

# Wait for all background processes to complete
wait
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "First Done at $(date +%Y-%m-%d_%H-%M-%S)"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"


CUDA_VISIBLE_DEVICES=0 python scripts/test_time_compute.py analysis/dispersion_toy_example.yaml  --temperature=0.8 --search_batch_size=1 > logs_ttt/slurm_4_%j.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/test_time_compute.py analysis/dispersion_toy_example.yaml  --temperature=1.0 --search_batch_size=1 > logs_ttt/slurm_5_%j.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/test_time_compute.py analysis/dispersion_toy_example.yaml  --temperature=1.2 --search_batch_size=1 > logs_ttt/slurm_6_%j.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/test_time_compute.py analysis/dispersion_toy_example.yaml  --temperature=1.4 --search_batch_size=1 > logs_ttt/slurm_7_%j.log 2>&1 &

# Wait for all background processes to complete
wait
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "Second Done at $(date +%Y-%m-%d_%H-%M-%S)"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"



CUDA_VISIBLE_DEVICES=0 python scripts/test_time_compute.py analysis/dispersion_toy_example.yaml  --temperature=1.6 --search_batch_size=1 > logs_ttt/slurm_8_%j.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/test_time_compute.py analysis/dispersion_toy_example.yaml  --temperature=1.8 --search_batch_size=1 > logs_ttt/slurm_9_%j.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/test_time_compute.py analysis/dispersion_toy_example.yaml  --temperature=2.0 --search_batch_size=1 > logs_ttt/slurm_10_%j.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/test_time_compute.py analysis/dispersion_toy_example.yaml  --temperature=2.2 --search_batch_size=1 > logs_ttt/slurm_11_%j.log 2>&1 &


# Wait for all background processes to complete
wait

echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"

echo "Third Done at $(date +%Y-%m-%d_%H-%M-%S)"

echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
echo "Total time taken: $(( $(date +%s) - start_time )) seconds"
echo "================================================"
echo "================================================"
echo "================================================"
echo "================================================"
