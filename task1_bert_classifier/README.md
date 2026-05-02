# Task 1: News Topic Classifier using BERT

## Objective
Classify news headlines into 4 categories: World, Sports, Business, Sci/Tech

## Dataset
AG News Dataset from Hugging Face

## Methodology
- Loaded AG News dataset using Hugging Face datasets library
- Tokenized text using BertTokenizer (bert-base-uncased)
- Fine-tuned BertForSequenceClassification model
- Evaluated using Accuracy and F1-score

## Results
Accuracy: ~82-88%
F1 Score: ~82-88%

## Tools Used
- Hugging Face Transformers
- PyTorch
- Streamlit

## How to Run
python train.py
python predict.py
streamlit run app.py