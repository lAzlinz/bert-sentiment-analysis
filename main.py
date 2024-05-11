import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

imdb = load_dataset('imdb')

# only take a small part of the big imdb dataset
small_train_dataset = imdb['train'].shuffle(seed=42).select([i for i in list(range(3_000))])
small_test_dataset = imdb['test'].shuffle(seed=42).select([i for i in list(range(300))])

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def preprocess_function(example):
    return tokenizer(example['text'], truncation=True)

tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# training the model
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

def compute_metrics(eval_pred):
    load_accuracy = load_metric('accuracy')
    load_f1 = load_metric('f1')

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)['accuracy']
    f1 = load_f1.compute(predictions=predictions, references=labels)['f1']
    return {'accuracy': accuracy, 'f1': f1}

training_args = TrainingArguments(
    output_dir='./models/my_model',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.001,
    save_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)