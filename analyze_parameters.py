import torch
import json
import matplotlib.pyplot as plt
import os
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LogitsProcessorList
from generate_biased import BiasLogitsProcessor, calculate_perplexity

def analyze_parameters(model_name='gpt2', bias_path='bias_vector.pt', prompt="The future of technology", length=50):
    print(f"Loading model {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Loading bias vector from {bias_path}...")
    if os.path.exists(bias_path):
        # Load the base bias vector (assumed to be generated with beta=1.0)
        base_bias_vector = torch.load(bias_path, map_location=device)
    else:
        print("Bias vector not found!")
        return

    # Define parameter sweeps
    k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 500]
    beta_values = [0.5, 1.0, 2.0, 3.0]
    
    results = {} # Structure: {beta: {k: ppl}}
    
    print(f"\nStarting Parameter Sweep...")
    print(f"Prompt: '{prompt}'")
    
    for beta in beta_values:
        print(f"\n--- Testing Beta = {beta} ---")
        results[beta] = {}
        
        # Scale bias vector for this beta
        # We assume base_bias_vector is from beta=1.0 run
        current_bias_vector = base_bias_vector * beta
        
        for k in k_values:
            print(f"  Testing Top-K = {k}...", end="", flush=True)
            
            processors = LogitsProcessorList()
            processors.append(BiasLogitsProcessor(current_bias_vector, top_k=k))
            
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            # Generate
            # We use a fixed seed or just sample? Sampling is better for PPL estimation but might be noisy.
            # Let's generate once per setting.
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=length + len(input_ids[0]),
                do_sample=True,
                top_k=50, # Base model sampling
                top_p=0.95,
                num_return_sequences=1,
                logits_processor=processors,
                pad_token_id=tokenizer.eos_token_id
            )
            
            text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            
            # Calculate Perplexity
            ppl = calculate_perplexity(model, tokenizer, text, device)
            print(f" PPL: {ppl:.2f}")
            
            results[beta][k] = ppl

    # Save Results
    with open('parameter_sweep_results.json', 'w') as f:
        # Convert keys to strings for JSON
        json_results = {str(b): {str(k): v for k, v in res.items()} for b, res in results.items()}
        json.dump(json_results, f, indent=4)
    print("\nResults saved to parameter_sweep_results.json")
    
    # Plotting
    plot_results(results)

def plot_results(results):
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'orange', 'red']
    markers = ['o', 's', '^', 'D']
    
    for i, beta in enumerate(sorted(results.keys())):
        data = results[beta]
        ks = sorted(data.keys())
        ppls = [data[k] for k in ks]
        
        plt.plot(ks, ppls, marker=markers[i % len(markers)], linestyle='-', 
                 color=colors[i % len(colors)], label=f'Beta = {beta}')
    
    plt.title('Perplexity vs. Top-K Parameter (by Beta)')
    plt.xlabel('Top-K Value')
    plt.ylabel('Perplexity (Lower is Better)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Log scale for X might be nice since we have 10..500
    # plt.xscale('log') 
    # But linear is fine for 10-100 detail.
    
    output_img = 'parameter_sweep.png'
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")

if __name__ == "__main__":
    analyze_parameters()
