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
def load_pairs(pair_file):
    with open(pair_file, 'r') as f:
        pairs = json.load(f)  # Format: list of [text1, text2, label]
    return pairs

class ContrastiveTextDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        sentence, key_string, label = self.pairs[idx]

        sent_enc1 = self.tokenizer(sentence, truncation=True, padding='max_length', 
                           max_length=self.max_length, return_tensors='pt')
        key_enc1 = self.tokenizer(key_string, truncation=True, padding='max_length', 
                          max_length=self.max_length, return_tensors='pt')
       
        return {
            'sentence_input_ids_1': sent_enc1['input_ids'].squeeze(0),
            'sentence_attention_mask_1': sent_enc1['attention_mask'].squeeze(0),
            'key_input_ids_1': key_enc1['input_ids'].squeeze(0),
            'key_attention_mask_1': key_enc1['attention_mask'].squeeze(0),
            'label_1': torch.tensor(label, dtype=torch.float),
        }

    def __len__(self):
        return len(self.pairs)

# encoder model
class ContrastiveModel(nn.Module):
    def __init__(self, base_model , hidden_dim=768):
        super().__init__()
        self.encoder = DebertaModel.from_pretrained(base_model)
        #MLP encoder 
        self.key_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
      
    def forward(self, sentence_ids, sentence_mask, key_ids, key_mask):
        sentence_emb = self.encoder(input_ids=sentence_ids, attention_mask=sentence_mask).last_hidden_state[:,0,:]
        # Get embedding from DeBERTa
        raw_key_emb = self.encoder(input_ids=key_ids, attention_mask=key_mask).last_hidden_state[:,0,:]
        # Pass through key encoder MLP
        key_emb = self.key_encoder(raw_key_emb)
        # combined_emb = torch.cat((sentence_emb, key_emb), dim=1)
        return sentence_emb, key_emb  # Return both embeddings for contrastive loss

# Main training loop
def train(model, dataloader, optimizer, epochs):
    model.train()
    history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
             # ——— Unpack pair #1 ———
            s1 = batch['sentence_input_ids_1'].to(device)
            m1 = batch['sentence_attention_mask_1'].to(device)
            k1 = batch['key_input_ids_1'].to(device)
            km1 = batch['key_attention_mask_1'].to(device)

            sent_emb, key_emb = model(s1, m1, k1, km1)  

            loss = nt_xent_loss(sent_emb, key_emb)  

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


    # Save plot
    plt.plot(range(1, epochs+1), history, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Contrastive Training Loss')
    plt.savefig(os.path.join(output_dir, f'loss_plot_{len(pairs)}_pairs_{epochs}_epochs.png'))
    print(f"Saved loss plot to {output_dir}/loss_plot_{len(pairs)}_pairs_{epochs}_epochs.png")

# Run training
if __name__ == "__main__":
    pair_file = "data\contrastive_Sentence_key_pairs.json"  # path to your pair file
    pairs = load_pairs(pair_file)
    pairs = pairs[:100]
    dataset = ContrastiveTextDataset(pairs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ContrastiveModel(base_model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    train(model, dataloader, optimizer, epochs=2)
    
    

