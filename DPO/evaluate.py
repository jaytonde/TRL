import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Update these to your Hugging Face Hub model repo IDs ---
sft_model_id = "your-username/sft-model-repo"
dpo_model_id = "your-username/dpo-model-repo"

# Load models and tokenizers once to avoid reloading on every inference
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # or "auto"
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

sft_model, sft_tokenizer = load_model_and_tokenizer(sft_model_id)
dpo_model, dpo_tokenizer = load_model_and_tokenizer(dpo_model_id)

def generate_response(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    
    # If your tokenizer supports apply_chat_template (e.g. from trl or custom), use it
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback: just use the prompt text directly
        text = prompt

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )

    # Remove prompt tokens from generated output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")

st.title("SFT vs. DPO Model Comparison Tool")

prompt = st.text_area("Enter your prompt:", height=100)

if st.button("Generate Responses"):
    if prompt.strip():
        col1, col2 = st.columns(2)

        with col1:
            st.header("SFT Model Output")
            with st.spinner("Generating SFT response..."):
                sft_response = generate_response(sft_model, sft_tokenizer, prompt)
                st.write(sft_response)

        with col2:
            st.header("DPO Model Output")
            with st.spinner("Generating DPO response..."):
                dpo_response = generate_response(dpo_model, dpo_tokenizer, prompt)
                st.write(dpo_response)
    else:
        st.warning("Please enter a prompt to generate responses.")
