import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from generate_biased import generate_text

# Configuration
PROMPTS_FILE = "prompts.md"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BIAS_PATH = "bias_qwen.pt"
REPETITION_PENALTIES = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
BETA = 1.0  # Fixed beta for this analysis
OUTPUT_CSV = "evaluation_results_repetition.csv"
OUTPUT_PLOT_PPL = "perplexity_vs_repetition.png"
OUTPUT_PLOT_RAP = "rap_vs_repetition.png"

def parse_prompts(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract numbered items (e.g., "1. Write a story...")
    prompts = re.findall(r'^\d+\.\s+(.+)$', content, re.MULTILINE)
    return prompts

def load_model():
    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        
    return model, tokenizer

def main():
    # 1. Load Prompts
    prompts = parse_prompts(PROMPTS_FILE)
    print(f"Found {len(prompts)} prompts. Running full evaluation...")
    
    # 2. Load Model
    model, tokenizer = load_model()
    
    results_data = []
    
    # 3. Evaluation Loop
    total_iterations = len(prompts) * len(REPETITION_PENALTIES)
    current_iter = 0
    
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        for rep_penalty in REPETITION_PENALTIES:
            current_iter += 1
            print(f"  [Progress {current_iter}/{total_iterations}] Repetition Penalty: {rep_penalty}")
            
            try:
                # Run generation
                res = generate_text(
                    model_name=MODEL_NAME,
                    bias_path=BIAS_PATH,
                    prompt=prompt,
                    length=100,
                    beta=BETA,
                    top_k_bias=50,
                    chat_mode=True,
                    repetition_penalty=rep_penalty,
                    model=model,
                    tokenizer=tokenizer
                )
                
                # Collect metrics
                results_data.append({
                    "Prompt ID": i + 1,
                    "Prompt": prompt,
                    "Repetition Penalty": rep_penalty,
                    "Biased Perplexity": res['biased']['perplexity'],
                    "RAP": res['biased']['prob'],
                    "Baseline Perplexity": res['baseline']['perplexity'],
                    "Baseline Author Score": res['baseline']['auth_score'],
                    "Biased Author Score": res['biased']['auth_score']
                })
                
            except Exception as e:
                print(f"  Error generating for prompt {i+1}, rep_penalty {rep_penalty}: {e}")

    # 4. Save Data
    df = pd.DataFrame(results_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")
    
    # 5. Generate Plots
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Perplexity vs Repetition Penalty
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Repetition Penalty", y="Biased Perplexity", marker="o")
    plt.title("Biased Model Perplexity vs. Repetition Penalty")
    plt.ylabel("Perplexity (Lower is Better)")
    plt.xlabel("Repetition Penalty")
    plt.savefig(OUTPUT_PLOT_PPL)
    print(f"Saved plot: {OUTPUT_PLOT_PPL}")
    
    # Plot 2: RAP vs Repetition Penalty
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Repetition Penalty", y="RAP", marker="o", color="green")
    plt.title("Relative Authorship Probability (RAP) vs. Repetition Penalty")
    plt.ylabel("RAP (%) (Higher is Better)")
    plt.xlabel("Repetition Penalty")
    plt.savefig(OUTPUT_PLOT_RAP)
    print(f"Saved plot: {OUTPUT_PLOT_RAP}")

if __name__ == "__main__":
    main()
