from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
dataset = load_dataset("ag_news")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# Training settings
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].select(range(2000)),  # small = faster
    eval_dataset=dataset["test"].select(range(500)),
    compute_metrics=compute_metrics,
)

# Run training
trainer.train()

# Show results
results = trainer.evaluate()
print("\n✅ RESULTS:", results)

# Save model
model.save_pretrained("model")
tokenizer.save_pretrained("model")
print("\n✅ Model saved to /model folder")