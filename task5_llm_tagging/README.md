# Task 5: Auto Tagging Support Tickets Using LLM

## Objective
Automatically tag support tickets into categories using a large language model.

## Dataset
Custom support ticket dataset (free-text)

## Methodology
- Used facebook/bart-large-mnli for zero-shot classification
- Applied few-shot learning techniques
- Output top 3 most probable tags per ticket
- Compared zero-shot vs few-shot performance

## Results
- Zero-shot classification works accurately without any training
- Top tag confidence typically above 50%

## Tools Used
- Hugging Face Transformers
- facebook/bart-large-mnli
- Streamlit

## How to Run
python tagger.py
streamlit run app.py