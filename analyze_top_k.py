import torch
import json
import matplotlib.pyplot as plt
import os
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LogitsProcessorList
from generate_biased import BiasLogitsProcessor, calculate_perplexity

def analyze_top_k(model_name='gpt2', bias_path='bias_vector.pt', prompt="The future of technology", length=50, beta=1.0):
    print(f"Loading model {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Loading bias vector from {bias_path}...")
    if os.path.exists(bias_path):
        bias_vector = torch.load(bias_path, map_location=device)
        bias_vector = bias_vector * beta
    else:
        print("Bias vector not found!")
        return

    k_values = [10, 30, 50, 100, 200, 500, 1000, 0] # 0 represents 'All' (no filtering)
    results = {}
    
    print(f"\nStarting Top-K Sweep (Beta={beta})...")
    print(f"Prompt: '{prompt}'")
    
    for k in k_values:
        print(f"\nTesting Top-K = {k}...")
        
        # Setup LogitsProcessor
        processors = LogitsProcessorList()
        # If k=0, we pass None or handle it in BiasLogitsProcessor. 
        # Our BiasLogitsProcessor takes top_k. If top_k is None or 0, it might need adjustment.
        # Let's check generate_biased.py implementation.
        # It checks: if self.top_k is not None and self.top_k > 0:
        # So passing 0 effectively disables the filtering (applies bias to ALL tokens).
        
        processors.append(BiasLogitsProcessor(bias_vector, top_k=k))
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Generate
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=length + len(input_ids[0]),
            do_sample=True,
            top_k=50, # Base model sampling top-k (keep this constant to isolate our bias top-k effect?)
                      # Actually, if we want to test OUR top-k bias, we should probably keep the base sampling consistent.
            top_p=0.95,
            num_return_sequences=1,
            logits_processor=processors,
            pad_token_id=tokenizer.eos_token_id
        )
        
        text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        print(f"Generated: {text[:50]}...")
        
        # Calculate Perplexity
        ppl = calculate_perplexity(model, tokenizer, text, device)
        print(f"Perplexity: {ppl:.2f}")
        
        results[k] = ppl

    # Save Results
    with open('top_k_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to top_k_results.json")
    
    # Plotting
    plot_results(results)

def plot_results(results):
    # Prepare data
    ks = list(results.keys())
    ppls = list(results.values())
    
    # Handle '0' (All) for plotting
    # We'll plot 0 as a separate point or just label it 'All' at the end?
    # Let's plot it as a bar chart or line chart. 
    # Since 0 is effectively "Max Vocab", it's hard to plot on a linear x-axis with 10, 50, etc.
    # Let's treat them as categorical categories for a bar chart, or just exclude 0 from the line and show it as a reference line.
    
    # Let's separate numeric K and 'All' (0)
    numeric_k = [k for k in ks if k != 0]
    numeric_ppl = [results[k] for k in numeric_k]
    
    all_ppl = results.get(0, None)
    
    plt.figure(figsize=(10, 6))
    
    # Plot line for numeric K
    plt.plot(numeric_k, numeric_ppl, marker='o', linestyle='-', label='Top-K Filtered')
    
    # Plot 'All' as a horizontal line if it exists
    if all_ppl is not None:
        plt.axhline(y=all_ppl, color='r', linestyle='--', label=f'No Filter (K=All): {all_ppl:.2f}')
    
    plt.title('Perplexity vs. Top-K Bias Parameter')
    plt.xlabel('Top-K Value (Number of tokens biased)')
    plt.ylabel('Perplexity (Lower is Better)')
    plt.grid(True)
    plt.legend()
    
    output_img = 'top_k_perplexity.png'
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")

if __name__ == "__main__":
    analyze_top_k()
