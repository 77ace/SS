import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaTokenizer, DebertaModel
from tqdm import tqdm
import json

# Configs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model = "microsoft/deberta-base-mnli"
tokenizer = DebertaTokenizer.from_pretrained(base_model)
output_dir = "contrastive_output"
os.makedirs(output_dir, exist_ok=True)
loss_fn = nn.BCEWithLogitsLoss()


def load_data(pair_file):
    with open(pair_file, 'r') as f:
        pairs = json.load(f)  
    return pairs

class ContrastiveTextDataset(Dataset):
    def __init__(self, data, tokenizer,key, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.key = key

    def __getitem__(self, idx):
        Wm_output, label, = self.data[idx]['Watermarked_output'], self.data[idx]['label'],

        enc = self.tokenizer(Wm_output, self.key, truncation=True, padding='max_length',
                                    max_length=self.max_length, return_tensors='pt')

        return {
            "input_ids": enc['input_ids'].squeeze(),
            "attention_mask": enc['attention_mask'].squeeze(),
            "label": torch.tensor(label, dtype=torch.float),
        }

    def __len__(self):
        return len(self.data)

# encoder model
class WMDetector(nn.Module):
    def __init__(self, encoder_path, base_model):
        super().__init__()
        # Load the trained encoder
        self.encoder = DebertaModel.from_pretrained(base_model)
        
        self.encoder.load_state_dict(torch.load(encoder_path))
        
        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # manually pool: take the [CLS] token embedding
            last_hidden = outputs.last_hidden_state   # shape [B, T, H]
            pooled_output = last_hidden[:, 0, :]      # shape [B, H]

        logits = self.classifier(pooled_output)
        return logits

def trainDetector(model, dataloader, optimizer, epochs=3):
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels.unsqueeze(1))

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()
        # Print average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
    
    # Save Detector 
    save_path = os.path.join(output_dir, "perkey_watermark_detector.pt")
    torch.save(model.state_dict(), save_path)



if __name__ == "__main__":
    #load pairs
    data_file = os.path.join("data", "classifier_train.json")  # path to your pair file
    data = load_data(data_file)
    data = data[:100]
    # Create dataset and dataloader
    dataset = ContrastiveTextDataset(data, tokenizer,key="I_am_doing_my_research")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # Initialize model and optimizer
    model_path = os.path.join("contrastive_output/contrastive_model.pt")
    model = WMDetector(model_path, base_model).to(device)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-5)
    # Train the model
    trainDetector(model, dataloader, optimizer, epochs=2)