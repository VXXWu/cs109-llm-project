import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from generate_biased import generate_text

# Configuration - DEFAULT SETTINGS
PROMPTS_FILE = "prompts.md"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BIAS_PATH = "bias_qwen.pt"
BETA = 1.25
REPETITION_PENALTY = 1.2
LENGTH = 100
OUTPUT_CSV = "evaluation_results_default.csv"
OUTPUT_PLOT_BOXPLOT = "boxplot_comparison.png"

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
    print(f"Found {len(prompts)} prompts. Running evaluation with default settings...")
    print(f"Settings: Beta={BETA}, RepPenalty={REPETITION_PENALTY}, Length={LENGTH}")
    
    # 2. Load Model
    model, tokenizer = load_model()
    
    results_data = []
    
    # 3. Evaluation Loop
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        try:
            # Run generation with default settings
            res = generate_text(
                model_name=MODEL_NAME,
                bias_path=BIAS_PATH,
                prompt=prompt,
                length=LENGTH,
                beta=BETA,
                top_k_bias=50,
                chat_mode=True,
                repetition_penalty=REPETITION_PENALTY,
                model=model,
                tokenizer=tokenizer
            )
            
            # Collect metrics
            results_data.append({
                "Prompt ID": i + 1,
                "Prompt": prompt,
                "Biased Perplexity": res['biased']['perplexity'],
                "Baseline Perplexity": res['baseline']['perplexity'],
                "RAP": res['biased']['prob'],
                "Biased Author Score": res['biased']['auth_score'],
                "Baseline Author Score": res['baseline']['auth_score']
            })
            
        except Exception as e:
            print(f"  Error generating for prompt {i+1}: {e}")

    # 4. Save Data
    df = pd.DataFrame(results_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")
    
    # 5. Generate Box Plots
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Perplexity Comparison (hide outliers)
    ppl_data = pd.DataFrame({
        'Perplexity': list(df['Baseline Perplexity']) + list(df['Biased Perplexity']),
        'Type': ['Baseline']*len(df) + ['Biased']*len(df)
    })
    sns.boxplot(data=ppl_data, x='Type', y='Perplexity', ax=axes[0], showfliers=False)
    axes[0].set_title('Perplexity: Baseline vs Biased')
    axes[0].set_ylabel('Perplexity (Lower is Better)')
    
    # Plot 2: Author Likelihood Score Comparison (hide outliers)
    auth_data = pd.DataFrame({
        'Author Likelihood Score': list(df['Baseline Author Score']) + list(df['Biased Author Score']),
        'Type': ['Baseline']*len(df) + ['Biased']*len(df)
    })
    sns.boxplot(data=auth_data, x='Type', y='Author Likelihood Score', ax=axes[1], showfliers=False)
    axes[1].set_title('Author Likelihood Score: Baseline vs Biased')
    axes[1].set_ylabel('Author Likelihood Score (Higher is Better)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_BOXPLOT, dpi=300)
    print(f"Saved plot: {OUTPUT_PLOT_BOXPLOT}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Baseline Perplexity: Mean={df['Baseline Perplexity'].mean():.2f}, Median={df['Baseline Perplexity'].median():.2f}")
    print(f"Biased Perplexity: Mean={df['Biased Perplexity'].mean():.2f}, Median={df['Biased Perplexity'].median():.2f}")
    print(f"Baseline Author Score: Mean={df['Baseline Author Score'].mean():.2f}, Median={df['Baseline Author Score'].median():.2f}")
    print(f"Biased Author Score: Mean={df['Biased Author Score'].mean():.2f}, Median={df['Biased Author Score'].median():.2f}")
    print(f"RAP: Mean={df['RAP'].mean():.2f}%, Median={df['RAP'].median():.2f}%")

if __name__ == "__main__":
    main()
