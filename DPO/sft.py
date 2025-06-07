import os
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser
)

from datasets import load_dataset
from peft import AutoPeftModelForCausalLM





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

def create_datasets(tokenizer, configs, seed=None):
    dataset = load_dataset(
        configs.dataset_name,
        data_dir        = configs.subset,
        split           = configs.split,
        use_auth_token  = True,
        num_proc        = configs.num_workers if not configs.streaming else None,
        streaming       = configs.streaming,
    )
    if configs.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(configs.size_valid_set)
        train_data = dataset.skip(configs.size_valid_set)
        train_data = train_data.shuffle(buffer_size=configs.shuffle_buffer, seed=seed)
    else:
        dataset    = dataset.train_test_split(test_size=0.005, seed=seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

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

def main(configs):

    ################################Load Model###############################
    bnb_configs = None
    if configs.use_bnb:
        bnb_configs = BitsAndBytesConfig{
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype= torch.bfloat16,
        }

    model = AutoPeftModelForCausalLM.from_pretrained(
        configs.model_name,
        quantization_config  = bnb_configs,
        device_map           = {"": Accerelator().local_process_index},
        trust_remote_code    = True,
        use_auth_token       = True
    )
    model.config.use_cache = False  #Turning of the KVcaching

    #############################Set Tokenizer################################
    
    tokenizer              = AutoTokenizer.from_pretrained(configs.model_name, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    #############################Dataset Preparation##########################
    
    train_dataset, eval_dataset = create_datasets(tokenizer, configs, seed=configs.seed)

    ###############################Train model################################
    trainer = SFTTrainer(
                            model            = model,
                            train_dataset    = train_dataset,
                            eval_dataset     = eval_dataset,
                            peft_config      = peft_config,
                            max_length       = None,
                            formatting_func  = prepare_sample_text,
                            processing_class = tokenizer,
                            args             = training_args,
                        )
    trainer.train()
    trainer.save_model(training_args.output_dir) #This saves only lora trained adapter.

    output_dir = ps.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)    #This save whole model weights.

    #############################Merge Adaptor and Model##########################
    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries
    def print(self):
        print(self.entries)


if __nam__=="__main__":

    with open(r"configs.yaml") as f:
        configs = yaml.safe_load(f)
    configs = Config(**configs)

    main(configs)