import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, logging
logging.set_verbosity_error()
import argparse
import os
import math

class BiasLogitsProcessor(LogitsProcessor):
    def __init__(self, bias_vector, top_k=None, hyphen_penalty=10.0, tokenizer=None):
        self.bias_vector = bias_vector
        self.top_k = top_k
        self.hyphen_penalty = hyphen_penalty
        # Try to find hyphen token id dynamically
        if tokenizer:
             self.hyphen_token_id = tokenizer.convert_tokens_to_ids("-")
             # Fallback if not found or if it maps to unk (some tokenizers don't have standalone hyphen)
             if self.hyphen_token_id == tokenizer.unk_token_id:
                 self.hyphen_token_id = 12 # Fallback to GPT-2 default, though risky for other models
        else:
            self.hyphen_token_id = 12

    def __call__(self, input_ids, scores):
        # scores is [batch_size, vocab_size]
        
        # Ensure bias_vector is on the same device as scores (move only once if possible, but here we check)
        if self.bias_vector.device != scores.device:
            self.bias_vector = self.bias_vector.to(scores.device)

        # Apply hyphen penalty
        if self.hyphen_penalty > 0 and self.hyphen_token_id is not None:
             # Check bounds
            if self.hyphen_token_id < scores.shape[-1]:
                scores[:, self.hyphen_token_id] -= self.hyphen_penalty
        
        if self.top_k is not None and self.top_k > 0:
            # Optimized implementation: Avoid creating full vocab-size mask
            
            # 1. Get top-k indices: [batch_size, k]
            top_k_vals, top_k_indices = torch.topk(scores, self.top_k, dim=-1)
            
            # 2. Gather the bias values corresponding to these indices: [batch_size, k]
            # bias_vector is [vocab_size], we expand it to match batch dimension implicitly via indexing
            relevant_biases = self.bias_vector[top_k_indices]
            
            # 3. Add the biases to the scores at the specific indices
            # scatter_add_ adds values to the tensor at the indices
            # We want: scores[b, index] += bias[index]
            scores.scatter_add_(-1, top_k_indices, relevant_biases)
            
            return scores
        else:
            return scores + self.bias_vector

def calculate_perplexity(model, tokenizer, text, device):
    encodings = tokenizer(text, return_tensors='pt').to(device)
    # Check for n_positions or max_position_embeddings
    if hasattr(model.config, 'n_positions'):
        max_length = model.config.n_positions
    elif hasattr(model.config, 'max_position_embeddings'):
        max_length = model.config.max_position_embeddings
    else:
        max_length = 1024 # Default fallback

    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if len(nlls) > 0:
        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()
    else:
        return 0.0

def generate_text(model_name='Qwen/Qwen2.5-0.5B-Instruct', bias_path='bias_qwen.pt', prompt="The quick", length=100, beta=1.25, top_k_bias=50, chat_mode=False, temperature=0.7, hyphen_penalty=10.0, repetition_penalty=1.0, model=None, tokenizer=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model is None or tokenizer is None:
        print(f"Loading model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
    else:
        # Ensure model is on the correct device
        if model.device.type != device.type:
            model.to(device)
    
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    # Load bias vector
    print(f"Loading bias vector from {bias_path}...")
    if os.path.exists(bias_path):
        bias_vector_raw = torch.load(bias_path, map_location=device)
        
        # Get model's output embedding size
        model_vocab_size = model.get_output_embeddings().weight.shape[0]
        
        # Resize bias vector if needed
        if bias_vector_raw.shape[0] != model_vocab_size:
             # Only warn if the difference is significant (more than just padding/special tokens)
             # Qwen vocab is 151643, output is 151936 (diff 293). This is normal padding.
             if abs(bias_vector_raw.shape[0] - model_vocab_size) > 1000:
                 print(f"Warning: Bias vector size {bias_vector_raw.shape[0]} != Model output size {model_vocab_size}. Resizing.")
             
             new_bias = torch.zeros(model_vocab_size, device=device)
             # Copy what we can
             min_size = min(bias_vector_raw.shape[0], model_vocab_size)
             new_bias[:min_size] = bias_vector_raw[:min_size]
             bias_vector_raw = new_bias
        
        bias_vector = bias_vector_raw * beta
        print(f"Applied beta scaling: {beta}")
    else:
        print("Bias vector not found! Generating without bias.")
        model_vocab_size = model.get_output_embeddings().weight.shape[0]
        bias_vector = torch.zeros(model_vocab_size).to(device)

    # Handle Chat Mode Prompt
    if chat_mode:
        try:
            # Try to use the tokenizer's built-in chat template
            messages = [{"role": "user", "content": prompt}]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"Chat Mode Enabled. Using model-specific chat template:\n{full_prompt}")
        except Exception:
            # Fallback for models without a chat template (e.g. some base models)
            # We use a more conversational format to avoid "benchmark" artifacts
            full_prompt = f"User: {prompt}\nAssistant:"
            print(f"Chat Mode Enabled (Fallback template). Full Prompt:\n{full_prompt}")
    else:
        full_prompt = prompt
        print(f"Generating text with prompt: '{full_prompt}' (Top-K Bias: {top_k_bias}, Length: {length}, Temp: {temperature})")

    # --- Helper for Generation ---
    def run_generation(use_bias=False):
        input_ids = tokenizer.encode(full_prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
        logits_processor = LogitsProcessorList()
        if use_bias and os.path.exists(bias_path):
            # Create bias processor
            bias_processor = BiasLogitsProcessor(bias_vector, top_k=top_k_bias, hyphen_penalty=hyphen_penalty, tokenizer=tokenizer)
            logits_processor.append(bias_processor)

        # Generate text
        output_sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=len(input_ids[0]) + length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            repetition_penalty=repetition_penalty if use_bias else 1.0
        )
        
        generated_sequence = output_sequences[0].tolist()
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        
        # Remove the prompt from the output for cleaner display
        # We need to decode the original input_ids to get the exact prompt string used for encoding
        decoded_prompt = tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)
        if text.startswith(decoded_prompt):
            text = text[len(decoded_prompt):]
        
        # Post-processing to stop at the last complete sentence if it cuts off
        if chat_mode:
             # Simple heuristic: cut off after the last punctuation mark
            last_punct = max(text.rfind('.'), text.rfind('?'), text.rfind('!'))
            if last_punct != -1:
                text = text[:last_punct+1]
                
        return text

    # Load log probs for scoring
    log_probs_path = 'log_probs.pt'
    if os.path.exists(log_probs_path):
        log_probs = torch.load(log_probs_path).to(device)
        probs = torch.exp(log_probs) # Convert to raw probabilities
    else:
        log_probs = None
        probs = None
        print("Warning: log_probs.pt not found. Cannot calculate Author Probability.")

    def calculate_author_likelihood(text):
        if log_probs is None or len(text.strip()) == 0:
            return 0.0, 0.0
        
        # Tokenize
        tokens = tokenizer.encode(text)
        if len(tokens) == 0:
            return 0.0, 0.0
            
        token_ids = torch.tensor(tokens).to(device)
        
        # Filter out token_ids that are out of bounds of log_probs
        # This happens if the model generates special tokens or if vocab size mismatch
        valid_mask = token_ids < len(log_probs)
        valid_token_ids = token_ids[valid_mask]
        
        if len(valid_token_ids) == 0:
            return 0.0, 0.0
        
        # Sum log probabilities (Log-Likelihood)
        token_log_probs = log_probs[valid_token_ids]
        log_likelihood = token_log_probs.sum().item()
        
        # Sum raw probabilities (Sigma Frequency)
        token_probs = probs[valid_token_ids]
        sum_prob = token_probs.sum().item()
        
        return log_likelihood, sum_prob

    # --- 1. Baseline Generation (Unbiased) ---
    print("\n" + "="*60)
    print(f"BASELINE (Unbiased {model_name})")
    print("="*60)
    text_base = run_generation(use_bias=False)
    print(text_base)
    
    ppl_base = 0.0
    auth_score_base = 0.0
    if len(text_base) > 0:
        ppl_base = calculate_perplexity(model, tokenizer, text_base, device)
        auth_score_base, _ = calculate_author_likelihood(text_base)
        print(f"\nPerplexity: {ppl_base:.2f}")
        print(f"Author Likelihood Score: {auth_score_base:.2f}")

    # --- 2. Biased Generation ---
    print("\n" + "="*60)
    print(f"BIASED (Beta={beta}, Top-K={top_k_bias})")
    print("="*60)
    text_biased = run_generation(use_bias=True)
    print(text_biased)
    
    ppl_biased = 0.0
    auth_score_biased = 0.0
    prob = 0.0
    
    if len(text_biased) > 0:
        ppl_biased = calculate_perplexity(model, tokenizer, text_biased, device)
        auth_score_biased, _ = calculate_author_likelihood(text_biased)
        print(f"\nPerplexity: {ppl_biased:.2f}")
        print(f"Author Likelihood Score: {auth_score_biased:.2f}")
        
        # Relative Authorship Probability
        diff = auth_score_biased - auth_score_base
        try:
            prob = 1 / (1 + math.exp(-diff))
        except OverflowError:
            prob = 1.0 if diff > 0 else 0.0
        print(f"Relative Authorship Probability: {prob*100:.2f}%")

    results = {
        "biased": {
            "text": text_biased,
            "perplexity": ppl_biased,
            "auth_score": auth_score_biased,
            "prob": prob * 100
        },
        "baseline": {
            "text": text_base,
            "perplexity": ppl_base,
            "auth_score": auth_score_base
        }
    }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help="Hugging Face model name")
    parser.add_argument('--bias_path', type=str, default='bias_qwen.pt', help="Path to bias vector file")
    parser.add_argument('--prompt', type=str, default="The quick")
    parser.add_argument('--length', type=int, default=150, help="Number of tokens to generate")
    parser.add_argument('--beta', type=float, default=1.0, help="Bias scale factor")
    parser.add_argument('--top_k_bias', type=int, default=50, help="Apply bias only to top K tokens")
    parser.add_argument('--chat', action='store_true', help="Enable Chat Mode (Q&A format)")
    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature (lower = less random)")
    parser.add_argument('--hyphen_penalty', type=float, default=10.0, help="Penalty for hyphen token to prevent run-on words")
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help="Penalty for repeated tokens (default: 1.2)")
    args = parser.parse_args()
    
    generate_text(model_name=args.model, bias_path=args.bias_path, prompt=args.prompt, length=args.length, beta=args.beta, top_k_bias=args.top_k_bias, chat_mode=args.chat, temperature=args.temperature, hyphen_penalty=args.hyphen_penalty, repetition_penalty=args.repetition_penalty)
