import os
import spacy
import json
from nltk.tokenize import word_tokenize
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import argparse
import nltk
import time
from src.Generation.sampling import hash_key_sampling_with_context_auto
from src.Generation.utils import extract_entities_and_pos, preprocess_text, split_sentences, look_up_with_cache
from src.Generation.randomization import randomize

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def process_target(pair, index, tokenizer, lm_model, Top_K, Final_K, threshold, hash_key):
    """
    Process each target word to generate replacements.
    """
    word = pair[1]
    try:
        # Find alternatives for the target word
        list_alternative = look_up_with_cache(pair[0], pair[1], index, tokenizer, lm_model, Top_K, Final_K, threshold)
        
        if not list_alternative:
            return None  # Skip if no valid alternatives found

        # Extract alternatives
        alternatives = [alt[0] for alt in list_alternative]
        similarity = [alt[1] for alt in list_alternative]

        if alternatives and similarity:
            # Hash-Based Sampling with Context
            selected_word = hash_key_sampling_with_context_auto(word, alternatives, similarity, hash_key, sentence=pair[0], position=index)
            print(f" Word: {word}")
            print(f" Alternatives: {alternatives}")
            print(f" Similarity: {similarity}")
            print(f" Sampled: {selected_word}")
            print("-------------------")
            return selected_word
        else:
            return word
    except Exception as e:
        print(f"Error processing target word '{word}': {e}")
        return word

def process_sentence(sentence, tokenizer, lm_model, Top_K, Final_K, threshold, hash_key):
    """
    Process a single sentence to find and replace words.
    """
    replacements = []
    
    # Extract target words for replacement
    sentence_target_pairs = extract_entities_and_pos(sentence)
    
    for sentence_text, target_word, index in sentence_target_pairs:
        # Process each target word
        replacement_word = process_target((sentence_text, target_word), index, tokenizer, lm_model, Top_K, Final_K, threshold, hash_key)
        
        if replacement_word and replacement_word != target_word:
            replacements.append((target_word, replacement_word))
    
    return replacements

def apply_replacements(sentence, replacements):
    """
    Apply the replacements to the sentence.
    """
    modified_sentence = sentence
    for original_word, replacement_word in replacements:
        # Use word boundaries to avoid partial word replacements
        import re
        pattern = r'\b' + re.escape(original_word) + r'\b'
        modified_sentence = re.sub(pattern, replacement_word, modified_sentence)
    
    return modified_sentence

def process_text(text, tokenizer, lm_model, Top_K, Final_K, threshold, hash_key):
    """
    Processes text to replace words while preserving the original format, including spaces and newlines.
    """
    lines = text.splitlines(keepends=True)  # Retain original newline characters
    final_text = []
    total_randomized_words = 0
    total_words = len(word_tokenize(text))

    for line in lines:
        if line.strip():  # Process non-empty lines
            replacements = []
            sentence_replacements = process_sentence(
                line.strip(), tokenizer, lm_model, Top_K, Final_K, threshold, hash_key
            )
            if sentence_replacements:
                replacements.extend(sentence_replacements)

            # Apply replacements to the original line
            if replacements:
                randomized_line = apply_replacements(line, replacements)
                final_text.append(randomized_line)
                total_randomized_words += len(replacements)
            else:
                final_text.append(line)  # Keep the original line
        else:
            final_text.append(line)  # Preserve empty lines

    # Combine all lines while preserving formatting
    return "".join(final_text), total_randomized_words, total_words

def watermark_single_text(text, hash_key="SafeSeal_Key_2024", top_k=15, final_k=3, threshold=0.8):
    """
    Apply SafeSeal watermarking to a single text input.
    
    Args:
        text (str): The input text to watermark
        hash_key (str): The hash key for watermarking
        top_k (int): Top K alternatives to consider
        final_k (int): Final K alternatives to select from
        threshold (float): Similarity threshold
    
    Returns:
        tuple: (watermarked_text, total_randomized_words, total_words)
    """
    # Initialize models
    print("Loading models...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    lm_model = RobertaForMaskedLM.from_pretrained('roberta-base', attn_implementation="eager")
    lm_model.eval()
    print("Models loaded successfully!")
    
    # Process the text
    start_time = time.time()
    watermarked_text, total_randomized_words, total_words = process_text(
        text, tokenizer, lm_model, top_k, final_k, threshold, hash_key
    )
    end_time = time.time()
    
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Total words: {total_words}")
    print(f"Randomized words: {total_randomized_words}")
    print(f"Randomization rate: {(total_randomized_words/total_words)*100:.2f}%")
    
    return watermarked_text, total_randomized_words, total_words

def main():
    parser = argparse.ArgumentParser(description='SafeSeal Single Text Watermarking')
    parser.add_argument('--text', type=str, required=True, help='Text to watermark')
    parser.add_argument('--hash_key', default='SafeSeal_Key_2024', type=str, help='Key for hash-based sampling')
    parser.add_argument('--top_k', default=15, type=int, help='Top K alternatives to consider')
    parser.add_argument('--final_k', default=3, type=int, help='Final K alternatives to select from')
    parser.add_argument('--threshold', default=0.8, type=float, help='Similarity threshold')
    parser.add_argument('--output', type=str, help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Apply watermarking
    watermarked_text, randomized_words, total_words = watermark_single_text(
        args.text, args.hash_key, args.top_k, args.final_k, args.threshold
    )
    
    # Print results
    print("\n" + "="*50)
    print("ORIGINAL TEXT:")
    print("="*50)
    print(args.text)
    print("\n" + "="*50)
    print("WATERMARKED TEXT:")
    print("="*50)
    print(watermarked_text)
    print("\n" + "="*50)
    print("STATISTICS:")
    print("="*50)
    print(f"Total words: {total_words}")
    print(f"Randomized words: {randomized_words}")
    print(f"Randomization rate: {(randomized_words/total_words)*100:.2f}%")
    
    # Save to file if output path is provided
    if args.output:
        result = {
            "original_text": args.text,
            "watermarked_text": watermarked_text,
            "total_words": total_words,
            "randomized_words": randomized_words,
            "randomization_rate": (randomized_words/total_words)*100,
            "hash_key": args.hash_key,
            "parameters": {
                "top_k": args.top_k,
                "final_k": args.final_k,
                "threshold": args.threshold
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")

if __name__ == '__main__':
    main() 