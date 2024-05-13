import torch, csv
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np, json

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

tokenized_train = train_set.map(preprocess_function, batched=True)
tokenized_eval = eval_set.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# training the model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # for binary
    # # return accuracy.compute(predictions=predictions, references=labels)
    # for multi-class
    accuracy = np.mean(predictions==labels)
    precision = precision_score(labels, predictions, average=None)
    recall = recall_score(labels, predictions, average=None)
    f1 = f1_score(labels, predictions, average=None)
    macro_f1 = f1_score(labels, predictions, average='macro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    conf_mat = confusion_matrix(labels, predictions)

    data: dict = {
        'epoch': 0,
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1-score': f1.tolist(),
        'f1-score-macro': macro_f1,
        'f1-score-weighted': weighted_f1,
        'confusion matrix': conf_mat.tolist()
    }

    path_file: str = './record/unbalanced_model_record.json'
    try:
        with open(path_file, 'r') as file:
            old_data: list[dir] = json.load(file)
            data['epoch'] = 1 + len(old_data)
            old_data.append(data)
    except FileNotFoundError:
        data['epoch'] = 1
        old_data = [data]
    
    with open(path_file, 'w') as file:
        json.dump(old_data, file, indent=4)

    return {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1}

training_args = TrainingArguments(
    output_dir='./models/unbalanced_model',
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
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)