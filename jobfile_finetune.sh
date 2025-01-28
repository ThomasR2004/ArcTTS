#!/bin/bash

#SBATCH --job-name=finetune_deepseek
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00

source venv/bin/activate
export HF_HOME=/scratch-shared/$USER/.cache_dir/
python finetunepython.py \
                          --pretrained_model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
                          --data_dir processed_arc_data.json \
                          --output_dir /scratch-shared/$USER/finetune_results/ \
                          --max_seq_length 8192 \
                          --per_device_train 8 \
                          --per_device_eval 8 \
                          --num_train_epochs 1 \
                          --optim adamw_8bit \
                          --bf16 \
                          --gradient_accumulation_steps 4 \
                          --packing \
                          --logging_steps 100