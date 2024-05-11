import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import ktrain

import pandas as pd

df = pd.read_csv('./datasets/100kDataset_withSentiment.csv')
