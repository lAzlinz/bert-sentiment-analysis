import torch, csv
from datasets import load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np

#pre-trained model name
model_name = 'bhadresh-savani/distilbert-base-uncased-emotion'
# model_name = 'distilbert-base-uncased'

# get the dataset
with open('./datasets/100kDataset_withSentiment.csv', encoding='utf-8', mode='r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    next(csv_reader)
    X = []
    y = []
    for row in csv_reader:
        X.append(row[0])
        y.append(int(row[1]))

# split the dataset into train, eval, and test
X_train_eval, X_test, y_train_eval, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train, X_eval, y_train, y_eval = train_test_split(X_train_eval, y_train_eval, train_size=0.7, stratify=y_train_eval, random_state=42)

# convert it to a dict with keys 'text' and 'label'
def convert_to_dict(X_col, y_col):
    return [{'text': X, 'label': y} for X, y in zip(X_col, y_col)]

train_set = convert_to_dict(X_train, y_train)
eval_set = convert_to_dict(X_eval, y_eval)
test_set = convert_to_dict(X_test, y_test)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(example):
    return tokenizer(example['text'], truncation=True)

tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# training the model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

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