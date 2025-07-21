

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaTokenizer, DebertaModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ——— Configs ———
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model = "microsoft/deberta-base-mnli"
tokenizer = DebertaTokenizer.from_pretrained(base_model)

# ——— Dataset for Inference ———
class DetectionDataset(Dataset):
    def __init__(self, test_file_path, tokenizer, key, max_length=256):
        with open(test_file_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.key = key
        self.max_length = max_length

    def __len__(self):
        return len(self.data) * 2

    def __getitem__(self, idx):
        pair_idx = idx // 2
        variant  = "pos" if idx % 2 == 0 else "neg"
        item     = self.data[pair_idx]

        enc = self.tokenizer(
            self.key,
            item[f"Watermarked_output_{variant}"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        label = item[f"label_{variant}"]

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }


# ——— Detector Model Definition ———
class WMDetector(nn.Module):
    def __init__(self, encoder_path, base_model):
        super().__init__()
        # 1) instantiate fresh DeBERTa
        self.encoder = DebertaModel.from_pretrained(base_model)
        # 2) load fine-tuned encoder-only weights
        state_dict = torch.load(encoder_path)
        self.encoder.load_state_dict(state_dict)
        # 3) freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 4) binary head (no sigmoid here—use BCEWithLogitsLoss semantics)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            last_hidden = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
            pooled = last_hidden[:, 0, :]   # [CLS] token
        return self.classifier(pooled)      # raw logits

# ——— Evaluation Loop ———
def evaluate(model, dataloader):

    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lab   = batch["label"].to(device)

            logits = model(ids, mask).squeeze(-1)      # shape [B]
            prob   = torch.sigmoid(logits)
            pred   = (prob > 0.5).float()

            preds.append(pred.cpu())
            labels.append(lab.cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

# ——— Main ———
if __name__ == "__main__":
    test_file= os.path.join( "data", "contrastive_pairs.json")    # must have 'text' & 'label'
    encoder_path= os.path.join("contrastive_output","contrastive_model.pt")
    detector_path = os.path.join("contrastive_output","perkey_watermark_detector.pt")
    key="I_am_doing_my_research"

    # prepare data
    dataset  = DetectionDataset(test_file, tokenizer, key)
    dataloader = DataLoader(dataset, batch_size=16)

    # load model
    detector = WMDetector(encoder_path, base_model).to(device)
    detector.load_state_dict(torch.load(detector_path))

    # run evaluation
    evaluate(detector, dataloader)
