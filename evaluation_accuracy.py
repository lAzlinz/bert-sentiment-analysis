import torch, csv
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np, json
from datasets import Dataset

#pre-trained model name
tokenizer_name = 'distilbert/distilbert-base-uncased'
model_name = './models/balanced_model/checkpoint-3063'
# model_name = 'distilbert-base-uncased'

# get the dataset
with open('./datasets/99k_balanced_dataset_with_sentiment.csv', encoding='utf-8', mode='r') as f:
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
def convert_to_dict_of_list(X_col, y_col):
    return {'text': [X for X in X_col], 'label': [y for y in y_col]}

train_set = convert_to_dict_of_list(X_train, y_train)
eval_set = convert_to_dict_of_list(X_eval, y_eval)
test_set = convert_to_dict_of_list(X_test, y_test)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# convert to datasets.Dataset
train_dataset = Dataset.from_dict(train_set)
eval_dataset = Dataset.from_dict(test_set)

def preprocess_function(example):
    return tokenizer(example['text'], truncation=True)

tokenized_train = train_dataset.map(preprocess_function)
tokenized_eval = eval_dataset.map(preprocess_function)

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

    path_file: str = './record/balanced_model_record.json'
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
    output_dir='./models/balanced_model/asd',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy='epoch',
    load_best_model_at_end=True
)

if __name__ == '__main__':
    balanced_checkpoints = [
        '3063',
        '6126',
        '9189',
        '12252',
        '15315',
        '18378',
        '21441',
        '24504',
        '27567',
        '30630'
    ]

    checkpoints = [f'./models/balanced_model/checkpoint-{balanced_checkpoint}' for balanced_checkpoint in balanced_checkpoints]

    for cp in checkpoints:
        model = AutoModelForSequenceClassification.from_pretrained(cp)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.evaluate()