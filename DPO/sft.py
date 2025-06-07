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
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=configs.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=configs.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset



def main():

    ################################Load Model###############################
    bnb_config = None
    if config.use_bnb:
        bnb_config = BitsAndBytesConfig{
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype= torch.bfloat16,
        }

    model = AutoPeftModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config  = bnb_config,
        device_map           = {"": Accerelator().local_process_index},
        trust_remote_code    = True,
        use_auth_token       = True
    )
    model.config.use_cache = False  #Turning of the KVcaching

    #############################Set Tokenizer################################
    
    tokenizer              = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    #############################Dataset Preparation##########################
    
    train_dataset, eval_dataset = create_datasets(tokenizer, configs, seed=configs.seed)


if __nam__=="__main__":
    main()