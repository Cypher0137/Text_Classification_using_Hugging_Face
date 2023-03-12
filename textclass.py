import pandas as pd
import numpy as np
from transformers import AutoTokenizer


df = pd.read_csv("yelp.csv")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(list(df['text']), truncation=True, padding=True)
labels = df['stars'].values

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))

from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

train_encodings, val_encodings, train_labels, val_labels = train_test_split(encodings, labels, test_size=1/5, random_state=0)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=(train_encodings, train_labels),
    eval_dataset=(val_encodings, val_labels)
)

trainer.train()


trainer.save_model('./saved_model')

from transformers import pipeline, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./saved_model')
model = AutoModelForSequenceClassification.from_pretrained('./saved_model')

text_classification = pipeline(
    'text-classification',
    model=model,
    tokenizer=tokenizer
)

input_text = "I had an amazing experience at this restaurant. The food was delicious, and the service was excellent. I would definitely give it a 5-star rating."
predicted_category = text_classification(input_text)


ac = accuracy_score(eval_dataset, input_text)
pr = precision_score(eval_dataset, input_text)
re = recall_score(eval_dataset, input_text)
f1 = f1_score(eval_dataset, input_text)

print("Accuracy : ",ac)
print("Precision : ",pr)
print("Recall : ",re)
print("F1 Score : ",f1)
