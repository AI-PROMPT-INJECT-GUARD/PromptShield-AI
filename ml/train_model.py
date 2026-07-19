import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
# Load processed datasets
train_df = pd.read_csv("dataset/processed/train.csv")
validation_df = pd.read_csv("dataset/processed/validation.csv")
test_df = pd.read_csv("dataset/processed/test.csv")

# Display basic information
print("Training Dataset Shape:", train_df.shape)
print("Validation Dataset Shape:", validation_df.shape)
print("Test Dataset Shape:", test_df.shape)
from transformers import DistilBertTokenizer

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

print("\nDistilBERT tokenizer loaded successfully!")
# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)

print("\nDatasets converted successfully!")

print("Training Dataset:", train_dataset)
print("Validation Dataset:", validation_dataset)
print("Test Dataset:", test_dataset)

# Function to tokenize text
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Tokenize all datasets
train_dataset = train_dataset.map(tokenize_function)
validation_dataset = validation_dataset.map(tokenize_function)
test_dataset = test_dataset.map(tokenize_function)

print("\nDatasets tokenized successfully!")

# Display the first tokenized sample
print("\nFirst Tokenized Sample:")
print(train_dataset[0])

# Load DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

print("\nDistilBERT model loaded successfully!")

# Configure training
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
)

print("\nTraining arguments configured successfully!")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary"
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

print("\nTrainer created successfully!")

# Start training
print("\nStarting model training...\n")

trainer.train()

print("\nModel training completed successfully!")

# Evaluate the model
print("\nEvaluating model...\n")

results = trainer.evaluate()

print("\nEvaluation Results:")
print(results)

# Save the trained model
trainer.save_model("./saved_model")

# Save the tokenizer
tokenizer.save_pretrained("./saved_model")

print("\nModel and tokenizer saved successfully!")