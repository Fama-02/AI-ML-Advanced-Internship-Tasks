import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("model")
tokenizer = BertTokenizer.from_pretrained("model")

labels = ["World", "Sports", "Business", "Sci/Tech"]

st.title("📰 News Topic Classifier (BERT)")
st.write("Enter a news headline below and the model will classify it.")

text = st.text_input("News Headline")

if st.button("Classify"):
    if text:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits).item()
        st.success(f"Category: **{labels[pred]}**")
    else:
        st.warning("Please enter a headline first.")