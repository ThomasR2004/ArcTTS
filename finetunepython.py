import os
import torch
from dataclasses import dataclass, field

from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import HfArgumentParser
from datasets import load_dataset

@dataclass
class ExperimentArguments:
    """
    Arguments for user experiments
    """
    pretrained_model_name_or_path: str = field(
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        metadata={"help": "HuggingFace model name or local checkpoint"}
    )
    data_dir: str = field(
        default="processed_arc_data.json",
        metadata={"help": "Path to dataset"}
    )
    from_foundation_model: bool = field(
        default=True,
        metadata={"help": "Start from a foundation model or an instruct model"}
    )

    def __post_init__(self):
        if not self.pretrained_model_name_or_path or not self.data_dir:
            raise ValueError("Please specify a model and dataset!")

def prepare_dataset(dataset, tokenizer, from_foundation_model=False):
    if from_foundation_model:
        tokenizer = get_chat_template(
            tokenizer,
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
            chat_template="chatml",
            map_eos_token=True,
        )

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=os.cpu_count() // 2)
    return dataset

def apply_qlora(model, max_seq_length):
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        max_seq_length=max_seq_length,
        random_state=47,
    )
    return model

def main(user_config, sft_config):
    # Enable multi-GPU training
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Load dataset
    dataset = load_dataset("json", data_files=user_config.data_dir, split="train")

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=user_config.pretrained_model_name_or_path,
        max_seq_length=sft_config.max_seq_length,
        device_map="auto",  # Automatically distribute across GPUs
        dtype=None,
        load_in_4bit=True,
    )

    dataset = prepare_dataset(dataset, tokenizer, user_config.from_foundation_model)
    sft_config.dataset_text_field = "text"

    # Apply QLoRA
    model = apply_qlora(model, sft_config.max_seq_length)

    # Use DistributedDataParallel (DDP) if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        ddp_find_unused_parameters=False  # Optimized for multi-GPU
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")

if __name__ == "__main__":
    parser = HfArgumentParser((ExperimentArguments, SFTConfig))
    user_config, sft_config = parser.parse_args_into_dataclasses()
    print(user_config, sft_config)
    main(user_config, sft_config)