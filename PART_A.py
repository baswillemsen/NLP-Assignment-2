# Import packages
import numpy as np
import pandas as pd
import random

# Import the data
train = pd.read_csv('data_folder/olid-train.csv')
test = pd.read_csv('data_folder/olid-test.csv')

# 1. Class distributions (1 point)
print(train['labels'].value_counts())
print(train['labels'].value_counts(normalize=True))
freq_0 = train['labels'].value_counts(normalize=True).iloc[0]
freq_1 = train['labels'].value_counts(normalize=True).iloc[1]
print(train[train['labels'] == 0].iloc[0]['text'])
print(train[train['labels'] == 1].iloc[0]['text'])

def evaluation(df, freq_0, freq_1):
    df['TP'] = (df['labels'] == 1) & (df['labels'] == df['predictions'])
    df['FN'] = (df['labels'] == 1) & (df['labels'] != df['predictions'])
    df['FP'] = (df['labels'] == 0) & (df['labels'] != df['predictions'])
    df['TN'] = (df['labels'] == 0) & (df['labels'] == df['predictions'])

    precision_1 = sum(df['TP']) / (sum(df['TP']) + sum(df['FP'])) if (sum(df['TP']) + sum(df['FP']) > 0) else 0
    precision_0 = sum(df['TN']) / (sum(df['FN']) + sum(df['TN'])) if (sum(df['FN']) + sum(df['TN']) > 0) else 0
    precision_avg = np.mean([precision_1, precision_0])
    precision_wavg = freq_0 * precision_0 + freq_1 * precision_1

    recall_1 = sum(df['TP']) / (sum(df['TP']) + sum(df['FN'])) if (sum(df['TP']) + sum(df['FN']) > 0) else 0
    recall_0 = sum(df['TN']) / (sum(df['FP']) + sum(df['TN'])) if (sum(df['TP']) + sum(df['FN']) > 0) else 0
    recall_avg = np.mean([recall_1, recall_0])
    recall_wavg = freq_0 * recall_0 + freq_1 * recall_1

    F1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1 > 0) else 0
    F1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0 > 0) else 0
    F1_avg = np.mean([F1_1, F1_0])
    F1_wavg = freq_0 * F1_0 + freq_1 * F1_1

    print('metric, class_1, class_0, avg, wavg')
    print("precision: ", precision_1, precision_0, precision_avg, precision_wavg)
    print("recall: ", recall_1, recall_0, recall_avg, recall_wavg)
    print("F1: ", F1_1, F1_0, F1_avg, F1_wavg)

# Random
df_random = test[['text', 'labels']]
predictions = []
for i in range(len(df_random)):
    if random.random() > 0.5:
        predictions.append(1)
    else:
        predictions.append(0)
predictions
df_random['predictions'] = predictions
evaluation(df_random, freq_0, freq_1)

# Majority
df = train[['text', 'labels']]
majority_class = df['labels'].value_counts().idxmax()
df_majority = test[['text', 'labels']]
predictions = [majority_class] * len(df_majority)
df_majority['predictions'] = predictions
evaluation(df_majority, freq_0, freq_1)