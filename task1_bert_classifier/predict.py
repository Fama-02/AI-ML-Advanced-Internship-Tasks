from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("model")
tokenizer = BertTokenizer.from_pretrained("model")

labels = ["World", "Sports", "Business", "Sci/Tech"]

text = input("Enter a news headline: ")
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits).item()

print(f"\nPredicted Category: {labels[pred]}")