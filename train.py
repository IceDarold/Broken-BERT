import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import tqdm

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cpu":
    torch.set_num_threads(8)

tokenizer = AutoTokenizer.from_pretrained("Ilseyar-kfu/broken_bert")
model = AutoModelForSequenceClassification.from_pretrained("Ilseyar-kfu/broken_bert")
model.to(device)

# 1. Analyze and Initialize word embeddings
weight = model.bert.embeddings.word_embeddings.weight.data
is_zero = (weight.pow(2).sum(dim=1) == 0)
non_zero_indices = torch.where(~is_zero)[0]
non_zero_mean = weight[non_zero_indices].mean(dim=0)
non_zero_std = weight[non_zero_indices].std()

print(f"Initializing {torch.sum(is_zero).item()} corrupted tokens.")

with torch.no_grad():
    model.bert.embeddings.word_embeddings.weight[is_zero] = non_zero_mean + torch.randn_like(weight[is_zero]) * non_zero_std * 0.1

# 2. Freeze everything except word_embeddings
for param in model.parameters():
    param.requires_grad = False
model.bert.embeddings.word_embeddings.weight.requires_grad = True

# 3. Data Loading
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

df_val = pd.read_csv("val_dataset.csv")
label_map = {'neutral': 0, 'positive': 1, 'negative': 2}
df_val["label_id"] = df_val["labels"].map(label_map)

MAX_LEN = 64
BATCH_SIZE = 64

val_encodings = tokenizer(df_val["text"].tolist(), truncation=True, padding=True, max_length=MAX_LEN)
train_dataset = SentimentDataset(val_encodings, df_val["label_id"].tolist())
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4. Training (Stage 1: Supervised on Val)
optimizer = AdamW(model.bert.embeddings.word_embeddings.parameters(), lr=1e-3)
epochs = 15
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss / len(train_loader)}")

# 5. Pseudo-labeling (Stage 2)
print("Starting pseudo-labeling...")
model.eval()
df_test = pd.read_csv("test.csv")
test_encodings = tokenizer(df_test["text"].tolist(), truncation=True, padding=True, max_length=MAX_LEN)

class TestDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings["input_ids"])

test_dataset = TestDataset(test_encodings)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

pseudo_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pseudo_labels.extend(torch.argmax(logits, dim=1).cpu().numpy())

# Combined training
combined_encodings = {k: val_encodings[k] + test_encodings[k] for k in val_encodings}
combined_labels = df_val["label_id"].tolist() + pseudo_labels
combined_dataset = SentimentDataset(combined_encodings, combined_labels)
combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.bert.embeddings.word_embeddings.parameters(), lr=5e-4)
epochs_pseudo = 5
total_steps = len(combined_loader) * epochs_pseudo
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

print("Starting combined training...")
model.train()
for epoch in range(epochs_pseudo):
    epoch_loss = 0
    for batch in combined_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
    print(f"Combined Epoch {epoch+1}/{epochs_pseudo} Loss: {epoch_loss / len(combined_loader)}")

# Final Prediction
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

inv_label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
ans = [inv_label_map[p] for p in test_preds]

submission = pd.DataFrame({"labels": ans, "id": df_test["id"]})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
