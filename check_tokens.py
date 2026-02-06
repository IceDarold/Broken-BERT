import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from collections import Counter

model_name = "Ilseyar-kfu/broken_bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
embeddings = model.bert.embeddings.word_embeddings.weight.data

special_tokens = [tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id]
special_token_names = [tokenizer.pad_token, tokenizer.unk_token, tokenizer.cls_token, tokenizer.sep_token, tokenizer.mask_token]

print("Special Tokens Status:")
for tid, name in zip(special_tokens, special_token_names):
    if tid is None: continue
    vec = embeddings[tid]
    is_zero = torch.all(vec == 0).item()
    print(f"{name} (ID {tid}): {'ZEROED' if is_zero else 'Intact'}")

# Load validation data to find frequent tokens
df_val = pd.read_csv("val_dataset.csv")
all_text = " ".join(df_val["text"].astype(str).tolist())
tokens = tokenizer.encode(all_text, add_special_tokens=False)
counts = Counter(tokens)

print("\nTop 20 Frequent Tokens Status:")
for tid, count in counts.most_common(20):
    vec = embeddings[tid]
    is_zero = torch.all(vec == 0).item()
    token_str = tokenizer.decode([tid])
    print(f"'{token_str}' (ID {tid}, count {count}): {'ZEROED' if is_zero else 'Intact'}")

