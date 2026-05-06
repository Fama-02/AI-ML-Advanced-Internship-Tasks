# 🤖 AI/ML Engineering – Advanced Internship Tasks
### DevelopersHub Corporation | Internship Submission

This repository contains the completed advanced tasks for the AI/ML Engineering Internship at DevelopersHub Corporation. Three out of five tasks were completed: Task 1, Task 2, and Task 5.

---

## 📁 Repository Structure

```
AI-ML-Advanced-Internship-Tasks/
│
├── task1_bert_classifier/        # News Topic Classifier Using BERT
│   ├── train.py
│   ├── app.py
│   └── README (see below)
│
├── task2_ml_pipeline/            # End-to-End ML Pipeline with Scikit-learn
│   ├── pipeline.py
│   ├── predict.py
│   ├── telco.csv
│   ├── model.pkl
│   └── README (see below)
│
├── task5_auto_tagging/           # Auto Tagging Support Tickets Using LLM
│   ├── tagger.py
│   ├── app.py
│   └── README (see below)
│
└── README.md
```

---

## ✅ Task 1: News Topic Classifier Using BERT

### Objective
Fine-tune a BERT transformer model to classify news headlines into topic categories using the AG News Dataset.

### Methodology / Approach
- Loaded and tokenized the **AG News Dataset** from Hugging Face Datasets
- Fine-tuned the `bert-base-uncased` pretrained model using Hugging Face Transformers
- Applied transfer learning techniques to adapt the model for 4-class news classification (World, Sports, Business, Sci/Tech)
- Evaluated the model using **Accuracy** and **F1-Score**
- Deployed an interactive demo using **Streamlit / Gradio** for live predictions

### Key Results
- Achieved high classification accuracy on the AG News test set
- F1-Score demonstrated balanced performance across all 4 topic categories
- Live demo allows users to input any news headline and get an instant topic prediction

### Skills Demonstrated
- NLP with Transformer models (BERT)
- Transfer learning & fine-tuning
- Evaluation metrics for text classification
- Lightweight model deployment

### Libraries Used
`transformers` `datasets` `torch` `scikit-learn` `streamlit`

---

## ✅ Task 2: End-to-End ML Pipeline with Scikit-learn

### Objective
Build a reusable, production-ready machine learning pipeline to predict customer churn using the Telco Churn Dataset.

### Methodology / Approach
- Loaded and explored the **Telco Churn Dataset** (`telco.csv`)
- Built a full preprocessing pipeline using `sklearn.pipeline.Pipeline`:
  - Numerical scaling with `StandardScaler`
  - Categorical encoding with `OneHotEncoder`
- Trained and compared two models: **Logistic Regression** and **Random Forest**
- Used **GridSearchCV** for hyperparameter tuning to find optimal parameters
- Exported the final trained pipeline using **joblib** (`model.pkl`) for reuse and deployment

### Key Results
- Random Forest outperformed Logistic Regression after hyperparameter tuning
- Pipeline is fully reusable — load `model.pkl` and run predictions on new data instantly
- Confusion matrix visualization saved for performance analysis

### Skills Demonstrated
- ML pipeline construction with Scikit-learn API
- Hyperparameter tuning with GridSearchCV
- Model export and reusability with joblib
- Production-readiness practices

### Libraries Used
`scikit-learn` `pandas` `numpy` `joblib` `matplotlib` `seaborn`

---

## ✅ Task 5: Auto Tagging Support Tickets Using LLM

### Objective
Automatically assign relevant tags to free-text support tickets using a Large Language Model (LLM) with prompt engineering and few-shot learning.

### Methodology / Approach
- Used a **Large Language Model (LLM)** to classify and tag support tickets
- Applied **prompt engineering** to guide the model towards accurate tagging
- Compared **zero-shot** vs **few-shot** performance to measure improvement
- Implemented **few-shot learning** by providing example ticket-tag pairs in the prompt
- Output the **top 3 most probable tags** per support ticket

### Key Results
- Few-shot prompting significantly improved tagging accuracy over zero-shot
- Model successfully identifies relevant categories such as: `billing`, `technical-issue`, `account`, `refund`, `connectivity`, etc.
- Top-3 tag output provides ranked predictions with confidence

### Skills Demonstrated
- Prompt engineering for LLMs
- Zero-shot and few-shot learning techniques
- LLM-based text classification
- Multi-class prediction and ranking

### Libraries Used
`openai` / `huggingface` `transformers` `pandas` `streamlit`

---

## 🛠️ How to Run

### Prerequisites
```bash
pip install transformers datasets torch scikit-learn pandas numpy joblib streamlit gradio
```

### Task 1 – BERT Classifier
```bash
cd task1_bert_classifier
python train.py        # Fine-tune the model
streamlit run app.py   # Launch interactive demo
```

### Task 2 – ML Pipeline
```bash
cd task2_ml_pipeline
python pipeline.py     # Train and export the pipeline
python predict.py      # Run predictions on new data
```

### Task 5 – Auto Tagging
```bash
cd task5_auto_tagging
python tagger.py       # Run the tagging script
streamlit run app.py   # Launch the demo (if applicable)
```

  

---

> ⭐ *Feel free to explore the code, and don't hesitate to raise issues or suggestions!*
