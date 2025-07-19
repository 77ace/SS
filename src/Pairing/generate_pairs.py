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

#generate_pairs('data/TRAIN_Mistral_top_3_ST_threshold_0.8_Uniform_0_10000_10000.json')

def generate_WM_Key_pairs(data_file, num_negative=300, key='I_am_doing_my_research'):
    with open(data_file, 'r') as f:
        samples = json.load(f)
        samples = samples[:300] # Limit to first 300 samples for demonstration  
    
    positive_pairs = [[record['Watermarked_output'], key , 1] for record in samples]
    negative_pairs = []
    for i in range(num_negative):
        record = random.choice(samples)
        fake_key = f"FakeKey_{random.randint(1000, 9999)}"
        negative_pairs.append([record['Watermarked_output'], fake_key , 0])  # [Watermarked output, fake key, 0]
    
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    # Save to file
    with open('data/contrastive_Sentence_key_pairs.json', 'w') as f:
        json.dump(all_pairs, f, indent=2)
    
    print(f"Saved {len(all_pairs)} total pairs to data/contrastive_Sentence_key_pairs.json")


#generate_WM_Key_pairs('data\Train_Llama2_top_3_threshold_0.8_KEY_I_am_doing_my_research_0_10k.json')

def generate_WM1_WM2_pairs(data_file_1, data_file_2, key1="I_am_doing_my_research", key2="This_is_my_test_key", num_samples=100):
    # Load positive samples
    with open(data_file_1, 'r') as f:
        samples_1 = json.load(f)[:num_samples]

    wm_1s = [sample['Watermarked_output'] for sample in samples_1]

    # Load negative samples
    with open(data_file_2, 'r') as f:
        samples_2 = json.load(f)[:num_samples]

    wm_2s = [sample['Watermarked_output'] for sample in samples_2]

    # Structure into labeled contrastive entries
    structured_pairs = []
    for i in range(num_samples):
        entry = {
            "Watermarked_output_pos": wm_1s[i],
            "Watermarked_output_neg": wm_2s[i],
            "key1": key1,
            "key2": key2,
            "label_pos": 1,
            "label_neg": 0
        }
        structured_pairs.append(entry)

    # Save to file
    output_path = 'data/contrastive_pairs.json'
    with open(output_path, 'w') as f:
        json.dump(structured_pairs, f, indent=2)

    print(f"Saved {len(structured_pairs)} structured pairs to {output_path}")

generate_WM1_WM2_pairs('data/Train_Llama2_top_3_threshold_0.8_KEY_I_am_doing_my_research_0_10k.json',"Generation_output/Train_Llama3_top_3_threshold_0.8_Key_This_is_my_test_key_100_200_100.json")