import streamlit as st
import torch
import os
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM
from generate_biased import generate_text
from analysis import analyze_corpus

# Page Config
st.set_page_config(page_title="Biased Text Generator", layout="wide")

# Title
st.title("Chat-GPMe")
st.markdown("Generate text that mimics a specific writing style using Logit Biasing. The model may take a couple of minutes to load.")

# Sidebar - Parameters
st.sidebar.header("Parameters")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
beta = st.sidebar.slider("Beta (Bias Strength)", 0.0, 5.0, 1.25, 0.1)
top_k_bias = st.sidebar.slider("Top-K Bias", 10, 200, 50, 10)
length = st.sidebar.slider("Generation Length", 50, 300, 100, 10)
temp = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
repetition_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 2.0, 1.2, 0.1)

# --- Model Caching ---
@st.cache_resource
def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    # Use low_cpu_mem_usage to avoid loading full model into RAM before sharding
    # Switch to float16 to reduce RAM usage (0.5B params * 2 bytes = ~1GB)
    model = AutoModelForCausalLM.from_pretrained(name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        
    return model, tokenizer

with st.spinner(f"Loading {model_name}..."):
    model, tokenizer = load_model(model_name)

# --- Main Interface ---

# 1. Corpus Selection
st.subheader("1. Select Style Corpus")
corpus_option = st.radio("Choose Corpus Source:", ["Default (Vince's Essays)", "Custom Text"])

bias_path = "bias_qwen.pt" # Default

if corpus_option == "Custom Text":
    custom_text = st.text_area("Paste your custom text here (the more the better):", height=200)
    if custom_text:
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt") as tmp:
            tmp.write(custom_text)
            tmp_path = tmp.name
        
        # Generate Bias
        if st.button("Analyze Custom Corpus"):
            with st.spinner("Analyzing corpus and generating bias vector..."):
                # We need to call analyze_corpus. 
                # Note: analyze_corpus saves to file. We can tell it to save to a temp bias file.
                custom_bias_path = tmp_path + ".pt"
                analyze_corpus(tmp_path, model_name=model_name, output_path=custom_bias_path)
                st.success("Analysis Complete! Custom bias vector generated.")
                bias_path = custom_bias_path
                # Store in session state to persist across reruns
                st.session_state['custom_bias_path'] = custom_bias_path

    # Use session state path if available
    if 'custom_bias_path' in st.session_state:
        bias_path = st.session_state['custom_bias_path']
        st.info(f"Using Custom Bias from: {bias_path}")

# 2. Prompt
st.subheader("2. Enter Prompt")
prompt = st.text_input("Prompt:", "Write a short story about a robot discovering a flower.")

# 3. Generate
if st.button("Generate Text", type="primary"):
    with st.spinner("Generating..."):
        results = generate_text(
            model_name=model_name,
            bias_path=bias_path,
            prompt=prompt,
            length=length,
            beta=beta,
            top_k_bias=top_k_bias,
            chat_mode=True,
            temperature=temp,
            repetition_penalty=repetition_penalty,
            model=model,
            tokenizer=tokenizer
        )
        
        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Unbiased Baseline")
            st.info(results['baseline']['text'])
            st.metric("Perplexity", f"{results['baseline']['perplexity']:.2f}")
            st.metric("Author Likelihood", f"{results['baseline']['auth_score']:.2f}")
            
        with col2:
            st.markdown(f"### Biased (Beta={beta})")
            st.success(results['biased']['text'])
            st.metric("Perplexity", f"{results['biased']['perplexity']:.2f}")
            st.metric("Author Likelihood", f"{results['biased']['auth_score']:.2f}")
            
            # Probability Metric
            prob = results['biased']['prob']
            st.metric("Relative Authorship Probability", f"{prob:.2f}%")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit & Qwen")
