import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import ktrain

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./datasets/100kDataset_withSentiment.csv')

classes = list(set(df.sentiment))

X_train, X_test, y_train, y_test = train_test_split(df.headline, df.sentiment, test_size=0.2, random_state=42)
