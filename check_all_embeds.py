import torch
from transformers import AutoModelForSequenceClassification

model_name = "Ilseyar-kfu/broken_bert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("Word Embeddings Zeros:", torch.sum(model.bert.embeddings.word_embeddings.weight == 0).item())
print("Position Embeddings Zeros:", torch.sum(model.bert.embeddings.position_embeddings.weight == 0).item())
print("Token Type Embeddings Zeros:", torch.sum(model.bert.embeddings.token_type_embeddings.weight == 0).item())

