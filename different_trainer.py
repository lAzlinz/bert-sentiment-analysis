import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
from ktrain import text

import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

df = pd.read_csv('./datasets/100kDataset_withSentiment.csv')

classes = list(set(df.sentiment))

X_train, X_test, y_train, y_test = train_test_split(df.headline, df.sentiment, test_size=0.2, random_state=42)

# preprocess
trn, val, preproc = text.texts_from_array(
    x_train=X_train.values,
    y_train=y_train.values,
    x_test=X_test.values,
    y_test=y_test.values,
    class_names=classes,
    val_pct=0.1, 
    max_features=30000, 
    maxlen=350,
    preprocess_mode='distilbert',
    ngram_range=1
)

# model
model = text.text_classifier('distilbert', train_data=trn, preproc=preproc)

# handling class imbalance
class_weights = class_weight.compute_class_weight('balanced', np.unique(df.sentiment), y_train)

weights = {}
for index, weight in enumerate(class_weights):
    weights[index] = weight

print(weights)