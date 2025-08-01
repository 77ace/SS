import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaTokenizer, DebertaModel
from sklearn.metrics import accuracy_score
import json
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.Losses.contrastive_losses import nt_xent_loss
#from src.Losses.Sup_contrastive_loss import sup_contrastive_loss

# Configs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model = "microsoft/deberta-base-mnli"
tokenizer = DebertaTokenizer.from_pretrained(base_model)
output_dir = "contrastive_output"
os.makedirs(output_dir, exist_ok=True)

# Load contrastive pairs (positive/negative) format = [WM_output, key_string, label]
def load_data(pair_file):
    with open(pair_file, 'r') as f:
        pairs = json.load(f)  # Format: list of [text1, text2, label]
    return pairs

class ContrastiveTextDataset(Dataset):
    def __init__(self, data, tokenizer, key1,key2, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.key1 = key1
        self.key2 = key2

    def __getitem__(self, idx):
        wm_output = self.data[idx]['Watermarked_output']
        enc1= self.tokenizer(wm_output, self.key1, truncation=True, padding='max_length', 
                            max_length=self.max_length, return_tensors='pt')
        enc2 = self.tokenizer(wm_output,self.key2, truncation=True, padding='max_length', 
                            max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': enc1['input_ids'].squeeze(0),
            'attention_mask': enc1['attention_mask'].squeeze(0),
            'input_ids_2': enc2['input_ids'].squeeze(0),
            'attention_mask_2': enc2['attention_mask'].squeeze(0),
        }
    
    def __len__(self):
        return len(self.data)

# encoder model
class ContrastiveModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = DebertaModel.from_pretrained(base_model)
        
    def forward(self, input_ids, att_mask):
        embedding = self.encoder(input_ids=input_ids, attention_mask=att_mask).last_hidden_state[:,0,:]
        return embedding  # Return both embeddings for contrastive loss

# Main training loop
def train(model, dataloader, optimizer, epochs):
    model.train()
    history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            
            ids_1 = batch['input_ids'].to(device)
            mask_1 = batch['attention_mask'].to(device)
            ids_2= batch['input_ids_2'].to(device)
            mask_2 = batch['attention_mask_2'].to(device)

            emb1 = model(ids_1, mask_1)
            emb2 = model(ids_2, mask_2)

            loss = nt_xent_loss(emb1, emb2)  

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        history.append(avg_loss)

    # Save model
    save_path = os.path.join(output_dir, "contrastive_model.pt")
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Model saved to {save_path}")


    # # Save plot
    # plt.plot(range(1, epochs+1), history, label='Train Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Contrastive Training Loss')
    # plt.savefig(os.path.join(output_dir, f'loss_plot_{len(data)}_pairs_{epochs}_epochs.png'))
    # print(f"Saved loss plot to {output_dir}/loss_plot_{len(data)}_pairs_{epochs}_epochs.png")

# Run training
if __name__ == "__main__":
    train_file = os.path.join("data","contrastive_Sentence_key_pairs.json")  # path to your pair file
    data = load_data(train_file)
    data = data[:100]
    key1 ="I_am_doing_my_research"
    key2= "This_is_my_test_key"
    dataset = ContrastiveTextDataset(data, tokenizer, key1=key1, key2=key2)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ContrastiveModel(base_model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    train(model, dataloader, optimizer, epochs=2)
    
    

