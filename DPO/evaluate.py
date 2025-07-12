import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Placeholder Functions for Model Inference ---
# Replace these with your actual model loading and inference functions

sft_model_path = "sft_model_output/final_merged_checkpoint"
dpo_model_path = "DPO_EXP_02/checkpoint-1000"

def get_sft_model_response(prompt):
    """
    Placeholder function for the SFT model.
    Takes a prompt and returns the model's response.
    """
    
    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype="auto",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    messages = [
                    {"role": "user", "content": prompt}
                ]

    text = tokenizer.apply_chat_template(
                                            messages,
                                            tokenize=False,
                                            add_generation_prompt=True
                                        )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
                                    **model_inputs,
                                    max_new_tokens=512
                                )

    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
    print(f"SFT model output : {generated_ids}")
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def get_dpo_model_response(prompt):
    """
    Placeholder function for the DPO model.
    Takes a prompt and returns the model's response.
    """
    model = AutoModelForCausalLM.from_pretrained(
        dpo_model_path,
        torch_dtype="auto",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(dpo_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    messages = [
                    {"role": "user", "content": prompt}
                ]

    text = tokenizer.apply_chat_template(
                                            messages,
                                            tokenize=False,
                                            add_generation_prompt=True
                                        )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
                                    **model_inputs,
                                    max_new_tokens=512
                                )

    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
    print(f"DPO model output : {generated_ids}")
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response


# --- Streamlit App Layout ---

st.set_page_config(layout="wide")

st.title("SFT vs. DPO Model Comparison Tool")

# Input prompt from the user
prompt = st.text_area("Enter your prompt:", height=100)

if st.button("Generate Responses"):
    if prompt:
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        with col1:
            st.header("SFT Model Output")
            with st.spinner("Generating SFT response..."):
                sft_response = get_sft_model_response(prompt)
                st.write(sft_response)

        with col2:
            st.header("DPO Model Output")
            with st.spinner("Generating DPO response..."):
                dpo_response = get_dpo_model_response(prompt)
                st.write(dpo_response)
    else:
        st.warning("Please enter a prompt to generate responses.")
