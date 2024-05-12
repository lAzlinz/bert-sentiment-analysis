import csv, pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

random_state = 42

with open('./datasets/100kDataset_withSentiment.csv', encoding='utf-8', mode='r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    labels = next(csv_reader)

    all = []
    for row in csv_reader:
        to_add = {}
        for index, label in enumerate(labels):
            try:
                value = int(row[index])
            except ValueError:
                value = row[index]
            
            to_add[label] = value
                
        
        all.append(to_add)


df = pd.DataFrame(all)

def get_class(class_value):
    return df[df['sentiment'] == class_value]

class_neg = get_class(0)
class_neu = get_class(1)
class_pos = get_class(2)

min_num_rows = min(len(class_neg), len(class_neu), len(class_pos))

def resampler(class_label):
    return resample(class_label, replace=False, n_samples=min_num_rows, random_state=random_state)


undersampled_neg = resampler(class_neg)
undersampled_neu = resampler(class_neu)
undersampled_pos = resampler(class_pos)

undersampled_df = pd.concat([undersampled_neg, undersampled_neu, undersampled_pos])

#  shuffle
undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# split the dataset into 3
X = undersampled_df['headline']
y = undersampled_df['sentiment']

# split the data to training and testing sets 7 : 3
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

# split the train_val data into training and validating set 7 : 3
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=0.7, random_state=random_state)

# the final split for train, val, and test are 0.49, 0.21, 0.3 respectively

#combine them back to dataframes
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# export them as CSVs

train_df.to_csv('balanced_train_dataset_with_sentiment_0.49.csv', index=False)
val_df.to_csv('balanced_val_dataset_with_sentiment_0.21.csv', index=False)
test_df.to_csv('balanced_test_dataset_with_sentiment_0.30.csv', index=False)