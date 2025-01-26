#!/bin/bash

#SBATCH --job-name=run_llm_api
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

# Load modules if required by SNELLIUS environment (adjust as needed)
module load python/3.9

# Activate virtual environment
source venv/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install openai vllm json os

# Set necessary environment variables
export HF_HOME=/scratch-shared/$USER/.cache_dir/
export OPENAI_API_KEY=BLANK

# Start the vLLM server (adjust parameters as necessary for your setup)
python -m vllm.entrypoints.api_server --model deepseek-r1-distill-qwen-32b & --model granite-3.1-8b-instruct


# Run your script
python arc.ipynb
