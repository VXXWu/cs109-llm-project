import torch
from transformers import AutoTokenizer
import numpy as np
import os
from collections import Counter
import math
import re
import argparse

def normalize_text(text):
    # Replace hyphens with spaces to split compound words
    text = text.replace('-', ' ')
    # Lowercase and remove punctuation
    text = text.lower()
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def analyze_corpus(corpus_path, model_name='gpt2', output_path='bias_vector.pt', epsilon=0.1, beta=1.0):
    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

    vocab_size = tokenizer.vocab_size
    
    print(f"Reading corpus from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Normalize corpus
    print("Normalizing corpus...")
    normalized_text = normalize_text(text)
    words = normalized_text.split()
    total_words = len(words)
    print(f"Total words in corpus (normalized): {total_words}")
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Map to tokenizer vocabulary
    print("Mapping to tokenizer vocabulary...")
    # We want to assign counts to tokens based on their normalized string representation
    token_counts = np.zeros(vocab_size, dtype=np.float64)
    
    for i in range(vocab_size):
        token_str = tokenizer.decode([i])
        norm_token = normalize_text(token_str).strip()
        
        if norm_token in word_counts:
            # Assign the count of the word to this token
            token_counts[i] = word_counts[norm_token]
            
    # 4. Define Dirichlet prior
    # alpha_v^(0) = epsilon
    alpha_0 = epsilon
    
    # 5. Compute posterior hyperparameters
    # alpha'_v = alpha_v^(0) + n_v
    alpha_prime = token_counts + alpha_0
            
    # 5.2 Compute total posterior mass
    alpha_prime_sum = np.sum(alpha_prime)
    
    # 6. Compute user's unigram distribution
    # theta_hat_v = alpha'_v / alpha'_0
    theta_hat = alpha_prime / alpha_prime_sum
    
    # 7. Convert to log-probabilities
    log_theta_hat = np.log(theta_hat)
    
    # CENTER THE BIAS VECTOR
    # We want frequent words to have positive bias and rare words to have negative bias.
    # Currently, all log-probs are negative. If we set stopwords to 0.0, they become the most probable.
    # We subtract the median to center the distribution.
    median_log_prob = np.median(log_theta_hat)
    print(f"Centering bias vector (Median log-prob: {median_log_prob:.4f})...")
    log_theta_hat_centered = log_theta_hat - median_log_prob
    
    # Save raw log probabilities for scoring (keep original uncentered for valid likelihood calculation)
    log_probs_tensor = torch.tensor(log_theta_hat, dtype=torch.float32)
    torch.save(log_probs_tensor, 'log_probs.pt')
    print("Saved raw log probabilities to log_probs.pt")
    
    # 8. Define log-prob bias scale using CENTERED values
    bias_vector = beta * log_theta_hat_centered
    
    # Identify stopwords and punctuation
    print("Identifying stopwords and punctuation tokens...")
    
    common_stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", 
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her", "its", "our", "their",
        "this", "that", "these", "those", "which", "who", "whom", "what", "where", "when", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
        "from", "up", "down", "out", "over", "under", "again", "further", "then", "once", "here", "there",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
        "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn",
        "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
        "shouldn", "wasn", "weren", "won", "wouldn", "me", "him", "us", "them", "myself", "yourself",
        "himself", "herself", "itself", "ourselves", "themselves", "during", "before", "after", "above",
        "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "through", "like",
        "about", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from"
    }
    
    # Add "junk" tokens that caused issues (e.g. "id" from "I'd")
    blacklist = {"id", "Id", "ID", "re", "ve", "ll", "m", "d"} 

    bias_vector = bias_vector.copy() # Use .copy() for numpy array
    
    zeroed_count = 0
    for i in range(vocab_size):
        token_raw = tokenizer.decode([i])
        token_norm = normalize_text(token_raw).strip()
        
        # Check if the NORMALIZED token is a stopword or empty
        if (token_norm in common_stopwords) or \
           (token_norm in blacklist) or \
           (not token_norm): # Empty after normalization (e.g. punctuation)
            
            bias_vector[i] = 0.0
            zeroed_count += 1
            
    print(f"Zeroing out bias for {zeroed_count} stopwords/punctuation/blacklist tokens...")
        
    # Convert to torch tensor
    bias_tensor = torch.tensor(bias_vector, dtype=torch.float32)
    
    print(f"Saving bias vector to {output_path}...")
    torch.save(bias_tensor, output_path)
    
    # Analysis stats
    print("\nAnalysis Stats:")
    print(f"Vocab size: {vocab_size}")
    print(f"Top 5 most frequent words in corpus:")
    most_common = word_counts.most_common(5)
    for word, count in most_common:
        print(f"  {word}: {count}")
        
    return bias_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='corpus.txt', help="Path to corpus file")
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help="Hugging Face model name for tokenizer")
    parser.add_argument('--output', type=str, default='bias_tinyllama.pt', help="Output path for bias vector")
    args = parser.parse_args()

    # Ensure corpus exists
    if not os.path.exists(args.corpus):
        print(f"Error: {args.corpus} not found.")
    else:
        analyze_corpus(args.corpus, model_name=args.model, output_path=args.output)
