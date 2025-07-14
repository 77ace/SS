import json
import random

def generate_pairs(data_file, num_negative=300):
    with open(data_file, 'r') as f:
        samples = json.load(f)
        samples = samples[:300] # Limit to first 300 samples for demonstration

    # Positive pairs: (original, watermarked) from same record
    positive_pairs = [[record['Original_output'], record['Watermarked_output'], 1] for record in samples]

    # Negative pairs: mismatched original + watermarked
    texts = [(record['Original_output'], record['Watermarked_output']) for record in samples]
    negative_pairs = []
    for _ in range(num_negative):
        record_A, record_B = random.sample(texts, 2)
        negative_pairs.append([record_A[0], record_B[1], 0])  # Original from one, Watermarked from another

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    # Save to file
    with open('data/contrastive_pairs.json', 'w') as f:
        json.dump(all_pairs, f, indent=2)

    print(f"Saved {len(all_pairs)} total pairs to data/contrastive_pairs.json")

generate_pairs('data/TRAIN_Mistral_top_3_ST_threshold_0.8_Uniform_0_10000_10000.json')