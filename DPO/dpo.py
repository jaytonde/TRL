import os
import logging
from typing import Optional
from dataclasses import dataclass, field

import torch
from peft import LoraConfig
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed

from trl import DPOConfig, DPOTrainer


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries
    def print(self):
        print(self.entries)

def get_training_args(configs):
    return DPOConfig(
        per_device_train_batch_size   = configs.per_device_train_batch_size,
        per_device_eval_batch_size    = configs.per_device_eval_batch_size,
        max_steps                     = configs.max_steps,
        logging_steps                 = configs.logging_steps,
        save_steps                    = configs.save_steps,
        gradient_accumulation_steps   = configs.gradient_accumulation_steps,
        gradient_checkpointing        = configs.gradient_checkpointing,
        learning_rate                 = configs.learning_rate,
        eval_strategy                 = "steps",
        eval_steps                    = configs.eval_steps,
        output_dir                    = configs.output_dir,
        report_to                     = configs.report_to,
        lr_scheduler_type             = configs.lr_scheduler_type,
        warmup_steps                  = configs.warmup_steps,
        optim                         = configs.optimizer_type,
        bf16                          = True,
        remove_unused_columns         = False,
        run_name                      = "dpo_llama2",
        gradient_checkpointing_kwargs = dict(use_reentrant=configs.gradient_checkpointing_use_reentrant),
        seed                          = configs.seed,
    )

def get_peft_configs(configs):
    return peft_config = LoraConfig(
        r              = configs.lora_r,
        lora_alpha     = configs.lora_alpha,
        lora_dropout   = configs.lora_dropout,
        target_modules = [
                            "q_proj",
                            "v_proj",
                            "k_proj",
                            "out_proj",
                            "fc_in",
                            "fc_out",
                            "wte",
                         ],
        bias           = "none",
        task_type      = "CAUSAL_LM",
    )

def main(configs):

    if configs.model_dtype == "float16":
        torch_dtype = torch.float16
        logger.info(f"model data types is : {torch_dtype}")
    elif configs.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
        logger.info(f"model data types is : {torch_dtype}")
    else:
        logger.info(f"Invalid data type : {configs.model_dtype}")

    logger.info("Loading base model for DPO.")
    model = AutoModelForCausalLM.from_pretrained(
                                                    configs.model_name_or_path,
                                                    low_cpu_mem_usage            = True,
                                                    torch_dtype                  = torch_dtype,
                                                    load_in_4bit                 = configs.load_in_4bit
                                                    device_map                   = {"": Accelerator().local_process_index}
                                                )
    model.config.use_cache = False
    logger.info("Model loaded successfully.")

    logger.info("Setting up the tokenizer.")
    tokenizer           = AutoTokenizer.from_pretrained(configs.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer setting completed.")

    logger.info("Loading the dataset train and eval set.")
    train_dataset = get_stack_exchange_paired(configs.train_data_dir)
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
                                beta              = script_args.beta,
                                train_dataset     = train_dataset,
                                eval_dataset      = eval_dataset,
                                processing_class  = tokenizer,
                                peft_config       = peft_config,
                                max_prompt_length = script_args.max_prompt_length,
                                max_length        = script_args.max_length,
                            )
    dpo_trainer.train()
    dpo_trainer.save_model(configs.output_dir)
    logger.info("Model training completed and adapter weights saved successfully.")

    logger.info("Saving whole model weights for later use.")
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
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
