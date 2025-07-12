import os
import yaml
import logging
from typing import Optional
from dataclasses import dataclass, field

import torch
from peft import LoraConfig
from accelerate import Accelerator
from datasets import Dataset, load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    is_torch_xpu_available, # This function checks if Intel's XPU (eXtensible Processing Unit) backend is available and properly configured for PyTorch.
    is_torch_npu_available,  # This function checks if Huawei's NPU (Neural Processing Unit) backend is available and properly configured for PyTorch.
    set_seed
)

from trl import DPOConfig, DPOTrainer

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries
    def print(self):
        print(self.entries)

def get_training_args(configs):
    return DPOConfig(
        num_train_epochs              = configs.num_train_epochs,
        per_device_train_batch_size   = configs.per_device_train_batch_size,
        per_device_eval_batch_size    = configs.per_device_eval_batch_size,
#        eval_strategy                 = configs.eval_strategy,
        max_steps                     = configs.max_steps,
        logging_steps                 = configs.logging_steps,
#        gradient_accumulation_steps   = configs.gradient_accumulation_steps,
#        gradient_checkpointing        = configs.gradient_checkpointing,
        learning_rate                 = configs.learning_rate,
        output_dir                    = configs.output_dir,
        report_to                     = configs.report_to,
        bf16                          = True
    )

def get_peft_configs(configs):
    return LoraConfig(
        r              = configs.lora_r,
        lora_alpha     = configs.lora_alpha,
        lora_dropout   = configs.lora_dropout,
        target_modules = configs.target_modules,
        bias           = configs.bias,
        task_type      = configs.task_type,
    )


def get_stack_exchange_paired(
    data_dir: str = "data/rl",
    cache_dir: Optional[str] = None,
    num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': list[str],
        'chosen': list[str],
        'rejected': list[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        "lvwerra/stack-exchange-paired",
        split=configs.split,
        #cache_dir=cache_dir,
        data_dir=data_dir,
        verification_mode="no_checks",
        token           = configs.hf_token
    )
    
    original_columns = dataset.column_names

    def return_prompt_and_responses(samples) -> dict[str, str]:
        return {
            "prompt": ["Question: " + question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

def main(logger, configs):

    mlflow.set_experiment("DPO Experiments")

    if configs.model_dtype == "float16":
        torch_dtype = torch.float16
        logger.info(f"model data types is : {torch_dtype}")
    elif configs.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
        logger.info(f"model data types is : {torch_dtype}")
    else:
        logger.info(f"Invalid data type : {configs.model_dtype}")

    logger.info("Loading base model for DPO.")

    logger.info("Setting BNB configs for model.")
    bnb_configs = None
    if configs.use_bnb:
        bnb_configs = BitsAndBytesConfig(
                                            load_in_4bit           = True,
                                            bnb_4bit_quant_type    = "nf4",
                                            bnb_4bit_compute_dtype = torch.bfloat16,
                                        )
    logger.info("BNB configs Set up completed.")

    logger.info("Loading base model for SFT.")
    model = AutoModelForCausalLM.from_pretrained(
                                                    configs.model_name_or_path,
                                                    quantization_config  = bnb_configs,
                                                    device_map           = {"": Accelerator().local_process_index},
                                                    trust_remote_code    = False
                                                )
    model.config.use_cache = False
    logger.info("Model loaded successfully.")

    logger.info("Setting up the tokenizer.")
    tokenizer           = AutoTokenizer.from_pretrained(configs.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer setting completed.")


    
    messages = [
                    {"role": "user", "content": "Hi There.."}
                ]

    messages2 = [
                    {"role": "user", "content": "Hi There how are you?"}
                ]

    text = tokenizer.apply_chat_template(
                                            messages,
                                            tokenize=False,
                                            add_generation_prompt=True
                                        )
    text2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)

    # Now, tokenize a batch of these texts
    batch_texts = [text, text2]

    model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(model.device)
    print(model_inputs['input_ids'])
    print(f"DPO tokenized : {tokenizer.decode(model_inputs['input_ids'].squeeze(0).cpu())}")


    

    logger.info("Loading the dataset train and eval set.")
    train_dataset = get_stack_exchange_paired()
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= configs.max_length
        and
        len(x["prompt"]) + len(x["chosen"]) <= configs.max_length,
        num_proc = configs.num_proc
    )

    eval_dataset = get_stack_exchange_paired(configs.eval_data_dir)
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= configs.max_length
        and
        len(x["prompt"]) + len(x["chosen"]) <= configs.max_length,
        num_proc = configs.num_proc
    )
    logger.info("Dataset loaded successfully.")

    logger.info("Loading Training ARGS and PEFT.")
    training_args = get_training_args(configs)
    peft_configs  = get_peft_configs(configs)
    logger.info("Configs loaded successfully.")

    logger.info("Model training started.")
    dpo_trainer = DPOTrainer(
                                model,
                                ref_model         = None,
                                args              = training_args,
                                train_dataset     = train_dataset,
                                #eval_dataset      = eval_dataset,
                                processing_class  = tokenizer,
                                peft_config       = peft_configs
                            )
    dpo_trainer.train()
    dpo_trainer.save_model(configs.output_dir)
    logger.info("Model training completed and adapter weights saved successfully.")

    logger.info("Saving whole model weights for later use.")
    output_dir = os.path.join(configs.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    logger.info("Model weights saved successfully.")

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading configurations from .yaml file")
    with open(r"configs_dpo.yaml") as f:
        configs = yaml.safe_load(f)
    configs = Config(**configs)
    logger.info("Loaded configurations successfully.")
    logger.info(f"SFT train configs : {configs}")

    main(logger, configs)
    logger.info("Model training completed successfully....")
