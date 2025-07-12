import os
import yaml
import wandb
import torch
import logging
from tqdm import tqdm

from accelerate import Accelerator

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    is_torch_xpu_available, # This function checks if Intel's XPU (eXtensible Processing Unit) backend is available and properly configured for PyTorch.
    is_torch_npu_available  # This function checks if Huawei's NPU (Neural Processing Unit) backend is available and properly configured for PyTorch.
)

from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig

from trl import SFTConfig, SFTTrainer
from trl.trainer import ConstantLengthDataset

from huggingface_hub import login

hf_token = ""
login(token=hf_token)

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries
    def print(self):
        print(self.entries)

def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

def create_datasets(logger, tokenizer, configs, seed=None):
    dataset = load_dataset(
        configs.dataset_name,
        data_dir        = configs.subset,
        split           = configs.split,
        num_proc        = configs.num_workers if not configs.streaming else None,
        streaming       = configs.streaming,
        token           = configs.hf_token
    )
    if configs.streaming:
        logger.info("Loading the dataset in streaming mode")
        valid_data = dataset.take(configs.size_valid_set)
        train_data = dataset.skip(configs.size_valid_set)
        train_data = train_data.shuffle(buffer_size=configs.shuffle_buffer, seed=seed)
    else:
        dataset    = dataset.train_test_split(test_size=0.005, seed=seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        logger.info(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    logger.info(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func  = prepare_sample_text,
        infinite         = True,
        seq_length       = configs.seq_length,
        chars_per_token  = chars_per_token,
    )
    
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func  = prepare_sample_text,
        infinite         = False,
        seq_length       = configs.seq_length,
        chars_per_token  = chars_per_token,
    )
    return train_dataset, valid_dataset


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model and their data types.
    """
    trainable_params = 0
    all_param = 0
    dtypes = set()
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            dtypes.add(param.dtype)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    print(f"Parameter data types in the model: {dtypes}")


def main(logger, configs):

    mlflow.set_experiment("DPO")

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
        configs.model_name,
        quantization_config  = bnb_configs,
        device_map           = {"": Accelerator().local_process_index},
        trust_remote_code    = False
    )
    model.config.use_cache = False  #Turning of the KVcaching
    logger.info("Model loaded successfully.")

    logger.info("Model trainable parameters.")
    print_trainable_parameters(model)

    logger.info("Setting up the tokenizer.")
    tokenizer              = AutoTokenizer.from_pretrained(configs.model_name, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("Tokenizer setting completed.")

    logger.info("Loading the dataset train and eval set.")
    train_dataset, eval_dataset = create_datasets(logger, tokenizer, configs, seed=configs.seed)
    logger.info("Dataset loaded successfully.")

    logger.info("Model training started.")
    training_args = SFTConfig(
        output_dir                  = configs.output_dir,
        max_steps                   = configs.max_steps,
        report_to                   = configs.report_to,
        num_train_epochs            = configs.num_train_epochs,
        per_device_train_batch_size = configs.per_device_train_batch_size,
        per_device_eval_batch_size  = configs.per_device_eval_batch_size,
        learning_rate               = configs.learning_rate,     
        max_seq_length              = configs.max_seq_length,
        dataset_text_field          = configs.dataset_text_field,
        #gradient_accumulation_steps = configs.gradient_accumulation_steps,
        bf16                        = configs.bf16,
        fp16                        = configs.fp16,
        #gradient_checkpointing      = configs.gradient_checkpointing,
        logging_steps               = configs.logging_steps
    )

    

    peft_config = LoraConfig(
                                r              = 8,
                                lora_alpha     = 16,
                                lora_dropout   = 0.05,
                                target_modules = ["q_proj", "v_proj"],
                                bias           = "none",
                                task_type      = "CAUSAL_LM",
                            )
    
    trainer = SFTTrainer(
                            model            = model,
                            train_dataset    = train_dataset,
                            eval_dataset     = eval_dataset,
                            peft_config      = peft_config,
                            formatting_func  = prepare_sample_text,
                            processing_class = tokenizer,
                            args             = training_args,
                        )
    trainer.train()
    trainer.save_model(training_args.output_dir) #This saves only lora trained adapter.
    logger.info("Model training completed and adapter weights saved successfully.")

    logger.info("Saving whole model weights for later use.")
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)    #This save whole model weights.
    logger.info("Model weights saved successfully.")

    del model
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


    logger.info("Merging base model with adapter.")
    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    logger.info("Merge and saved successfylly for direct inference.")

if __name__=="__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading configurations from .yaml file")
    with open(r"configs_sft.yaml") as f:
        configs = yaml.safe_load(f)
    configs = Config(**configs)
    logger.info("Loaded configurations successfully.")
    logger.info(f"SFT train configs : {configs.__dict__}")

    main(logger, configs)
    logger.info("Model training completed successfully....")
