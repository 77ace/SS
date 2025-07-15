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

#from src.Losses.contrastive_losses import nt_xent_loss
from src.Losses.Sup_contrastive_loss import sup_contrastive_loss

# Configs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model = "microsoft/deberta-base-mnli"
tokenizer = DebertaTokenizer.from_pretrained(base_model)
output_dir = "contrastive_output"
os.makedirs(output_dir, exist_ok=True)

# Load contrastive pairs (positive/negative)
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
        text1, text2, label = self.pairs[idx]
        enc1 = self.tokenizer(text1, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        enc2 = self.tokenizer(text2, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids_1': enc1['input_ids'].squeeze(),
            'attention_mask_1': enc1['attention_mask'].squeeze(),
            'input_ids_2': enc2['input_ids'].squeeze(),
            'attention_mask_2': enc2['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }

    def __len__(self):
        return len(self.pairs)

# encoder model
class ContrastiveModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = DebertaModel.from_pretrained(base_model)
      
    def forward(self, input_ids1, mask1, input_ids2, mask2):
        emb1 = self.encoder(input_ids=input_ids1, attention_mask=mask1).last_hidden_state[:,0,:]
        emb2 = self.encoder(input_ids=input_ids2, attention_mask=mask2).last_hidden_state[:,0,:]
        return emb1, emb2

# Main training loop
def train(model, dataloader, optimizer, epochs):
    model.train()
    history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            ids1 = batch['input_ids_1'].to(device)
            mask1 = batch['attention_mask_1'].to(device)
            ids2 = batch['input_ids_2'].to(device)
            mask2 = batch['attention_mask_2'].to(device)
            label = batch['label'].to(device)

            emb1, emb2 = model(ids1, mask1, ids2, mask2)
            loss = sup_contrastive_loss(emb1, emb2, label)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        history.append(avg_loss)

    # Save plot
    plt.plot(range(1, epochs+1), history, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Contrastive Training Loss')
    plt.savefig(os.path.join(output_dir, f'loss_plot_{len(pairs)}_pairs_{epochs}_epochs.png'))
    print(f"Saved loss plot to {output_dir}/loss_plot_{len(pairs)}_pairs_{epochs}_epochs.png")

# Run training
if __name__ == "__main__":
    pair_file = "data/contrastive_pairs.json"  # path to your pair file
    pairs = load_pairs(pair_file)
    pairs = pairs[:200]
    dataset = ContrastiveTextDataset(pairs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ContrastiveModel(base_model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    train(model, dataloader, optimizer, epochs=1)

