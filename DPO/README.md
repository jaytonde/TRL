Here's a `README.md` file in Markdown format for your DPO pipeline repository:

-----

# DPO Pipeline for Large Language Models

This repository contains code for fine-tuning a Large Language Model (LLM) using Supervised Fine-Tuning (SFT) and then further aligning it with human preferences using Direct Preference Optimization (DPO).

## Project Structure

The repository is organized as follows:

  * `configs_dpo.yaml`: Configuration file for the DPO training script.
  * `configs_sft.yaml`: Configuration file for the SFT training script.
  * `dpo.py`: Python script for DPO (Direct Preference Optimization) training.
  * `sft.py`: Python script for SFT (Supervised Fine-Tuning) training.
  * `evaluate.py`: A Streamlit application to compare responses from the SFT and DPO models.

## Setup

### Prerequisites

  * Python 3.8+
  * `pip` package installer

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install the required dependencies:**

    It's recommended to use a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt # You'll need to create a requirements.txt file based on the imports in sft.py and dpo.py
    ```

    *Self-correction:* The user has not provided a `requirements.txt` file. I should suggest they create one based on the imports in their scripts.

    Here are the likely dependencies you'll need in your `requirements.txt`:

    ```
    torch
    transformers
    accelerate
    datasets
    peft
    trl
    pyyaml
    mlflow
    huggingface_hub
    streamlit
    tqdm
    ```

    You might need specific versions of these libraries depending on your environment and hardware. For example, for `bitsandbytes` and `accelerate`, you may need to follow specific installation instructions from their respective documentation.

## Usage

### 1\. Supervised Fine-Tuning (SFT)

The SFT step fine-tunes the base model on a given dataset.

To run the SFT training, execute the `sft.py` script:

```bash
python sft.py
```

This will:

  * Load configurations from `configs_sft.yaml`.
  * Load the base model (`models/qwen2.5_3B`).
  * Train the model using the `lvwerra/stack-exchange-paired` dataset.
  * Save the SFT adapter weights and the merged model checkpoint to `./sft_model_output`.
  * Log training metrics to MLflow (assuming an MLflow server is running on `http://127.0.0.1:8080`).

### 2\. Direct Preference Optimization (DPO)

After SFT, the DPO step further refines the model's behavior based on preferred and rejected responses.

To run the DPO training, execute the `dpo.py` script:

```bash
python dpo.py
```

This will:

  * Load configurations from `configs_dpo.yaml`.
  * Load the SFT-trained model (`sft_model_output/final_merged_checkpoint`) as the base model for DPO.
  * Train the model using the `lvwerra/stack-exchange-paired` dataset (a subset defined in `configs_dpo.yaml`).
  * Save the DPO adapter weights and the full model checkpoint to `./DPO_EXP_03`.
  * Log training metrics to MLflow (assuming an MLflow server is running on `http://127.0.0.1:5000`).

### 3\. Model Evaluation

You can use the Streamlit application to interactively compare the responses of the SFT and DPO models.

To launch the evaluation app:

```bash
streamlit run evaluate.py
```

This will open a web application in your browser where you can:

  * Enter a prompt.
  * See the generated responses from both the SFT model (loaded from `sft_model_output/final_merged_checkpoint`) and the DPO model (loaded from `DPO_EXP_02/checkpoint-1000`).

## Configuration Files

### `configs_sft.yaml`

This file contains parameters for the Supervised Fine-Tuning phase:

  * `model_name`: Path to the base model.
  * `dataset_name`: Hugging Face dataset for SFT.
  * `subset`: Data subdirectory within the dataset.
  * `split`: Dataset split to use for training.
  * `size_valid_set`: Number of samples for the validation set.
  * `seq_length`: Maximum sequence length for training.
  * `use_bnb`: Whether to use BitsAndBytes for 4-bit quantization.
  * `seed`: Random seed for reproducibility.
  * `max_steps`: Maximum number of training steps.
  * `hf_token`: Hugging Face token for private datasets (if applicable).
  * **LoRA Parameters**: `r`, `lora_alpha`, `lora_dropout`, `target_modules`, `bias`, `task_type` for PEFT (Parameter-Efficient Fine-Tuning).
  * `output_dir`: Directory to save SFT model outputs.
  * `per_device_train_batch_size`, `per_device_eval_batch_size`: Batch sizes for training and evaluation.
  * `learning_rate`: Learning rate for the optimizer.
  * `num_train_epochs`: Number of training epochs.
  * `gradient_accumulation_steps`: Number of updates steps to accumulate before performing a backward/update pass.
  * `bf16`, `fp16`: Enable bfloat16 or float16 precision.
  * `gradient_checkpointing`: Enable gradient checkpointing.
  * `report_to`: Reporting tool (e.g., "mlflow").
  * `logging_steps`: Log training metrics every N steps.

### `configs_dpo.yaml`

This file contains parameters for the Direct Preference Optimization phase:

  * `model_name_or_path`: Path to the SFT-trained model.
  * `tokenizer_path`: Path to the tokenizer.
  * `dataset_name`: Hugging Face dataset for DPO.
  * `split`: Dataset split for DPO training (e.g., "train[:16000]").
  * `max_length`: Maximum sequence length for DPO.
  * `eval_data_dir`: Directory for evaluation data (though currently commented out in `dpo.py`).
  * `model_dtype`: Model data type (e.g., 'bfloat16').
  * `use_bnb`: Whether to use BitsAndBytes for 4-bit quantization.
  * `num_proc`: Number of processes for dataset mapping.
  * `hf_token`: Hugging Face token.
  * **SFT Configs (reused for DPO training arguments)**: `num_train_epochs`, `per_device_train_batch_size`, `per_device_eval_batch_size`, `max_steps`, `logging_steps`, `learning_rate`, `output_dir`, `report_to`, `bf16`.
  * **PEFT parameters**: `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`, `bias`, `task_type` for PEFT.

-----