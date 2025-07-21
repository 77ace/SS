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

def generate_Contrastive_train(data_file, num_negative=300, key='I_am_doing_my_research', fake_key='This_is_my_test_key'):
    with open(data_file, 'r') as f:
        samples = json.load(f)
        samples = samples[:300] # Limit to first 300 samples for demonstration  
    
    positive_pairs = [{
        "Watermarked_output":record['Watermarked_output'],
        "key": key , 
        'label':1} for record in samples
        ]
    
    negative_pairs = [
        {
        "Watermarked_output":record['Watermarked_output'],
        "key": fake_key , 
        'label':0} for record in samples
    ]
    
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    # Save to file
    with open('data/contrastive_train.json', 'w') as f:
        json.dump(all_pairs, f, indent=2)
    
    print(f"Saved {len(all_pairs)} total pairs to data/contrastive_train.json")


#generate_Contrastive_train('data/Train_Llama2_top_3_threshold_0.8_KEY_I_am_doing_my_research_0_10k.json')

def generate_classifier_train(data_file_1, data_file_2, key1="I_am_doing_my_research", key2="This_is_my_test_key", num_samples=100):
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
        pos_entry = {
            "Watermarked_output": wm_1s[i],
            "key": key1,
            "label": 1
        }
        neg_entry = {
            "Watermarked_output": wm_2s[i],
            "key": key2,
            "label": 0
        }
        structured_pairs.append(pos_entry)
        structured_pairs.append(neg_entry)

    random.shuffle(structured_pairs)
    # Save to file
    output_path = 'data/classifier_train.json'
    with open(output_path, 'w') as f:
        json.dump(structured_pairs, f, indent=2)

    print(f"Saved {len(structured_pairs)} structured pairs to {output_path}")

generate_classifier_train('data/Train_Llama2_top_3_threshold_0.8_KEY_I_am_doing_my_research_0_10k.json',"Generation_output/Train_Llama3_top_3_threshold_0.8_Key_This_is_my_test_key_100_200_100.json")