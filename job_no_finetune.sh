#!/bin/bash

#SBATCH --job-name=run_llm_api
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

# Load modules if required by SNELLIUS environment (adjust as needed)
module load 2023

# Activate virtual environment
source venv/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install openai vllm json os

# Set necessary environment variables
export HF_HOME=/scratch-shared/$USER/.cache_dir/
export OPENAI_API_KEY=BLANK

# Run the first vLLM server for ModelA on port 8000
vllm serve deepseek-r1-distill-qwen-32b --port 8000 --api-key token-1 &

# Run the second vLLM server for ModelB on port 8001
vllm serve granite-3.1-8b-instruct --port 8001 --api-key token-2 &


# Run your script
python arc.ipynb
