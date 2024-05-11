import torch
from datasets import load_dataset
from transformers import AutoTokenizer

imdb = load_dataset('imdb')

# only take a small part of the big imdb dataset
small_train_dataset = imdb['train'].shuffle(seed=42).select([i for i in list(range(3_000))])
small_test_dataset = imdb['test'].shuffle(seed=42).select([i for i in list(range(300))])

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def preprocess_function(example):
    return tokenizer(example['text'], truncation=True)

tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)