# -*- coding: utf-8 -*-
"""Part_A_and_B.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uyszc2DkQ0qitBPhgW8YNBQLZxIwA4xg

Install and import packages, mount **drive**
"""

!pip install --user simpletransformers

!pip install --user transformers

!sudo pip install checklist --user

!pip install --user tabulate

!pip install --user spacy

import numpy as np
import pandas as pd
import torch
import os
import logging
import spacy
import checklist
from checklist.perturb import Perturb
from checklist.editor import Editor
editor = Editor()
import random
random.seed(0)

from simpletransformers.classification import ClassificationModel, ClassificationArgs

from google.colab import drive
drive.mount('/content/gdrive/')

"""# 0. Preparation and definitions

**Import the data**
"""

# # Import the data
# train = pd.read_csv('../input/olid-train.csv')
# test = pd.read_csv('../input/olid-test.csv')
# subset_test = pd.read_csv('../input/olid-subset-diagnostic-tests.csv')

# Import the data
train = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/olid-train.csv')
test = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/olid-test.csv')
subset_test = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/olid-subset-diagnostic-tests.csv')

"""**Making the evaluation call; precision, recall and F1**"""

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

"""# 1. Class distributions (1 point)"""

# 1. Class distributions (1 point)
print(train['labels'].value_counts())
print(train['labels'].value_counts(normalize=True))
freq_0 = train['labels'].value_counts(normalize=True).iloc[0]
freq_1 = train['labels'].value_counts(normalize=True).iloc[1]
print(train[train['labels'] == 0].iloc[0]['text'])
print(train[train['labels'] == 1].iloc[0]['text'])

"""# 2.	Baselines (1 point) """

# Random
df_random = test[['text', 'labels']]
predictions = []
for i in range(len(df_random)):
    if random.random() > 0.5:
        predictions.append(1)
    else:
        predictions.append(0)
df_random['predictions'] = predictions
evaluation(df_random, freq_0, freq_1)

# Majority
df = train[['text', 'labels']]
majority_class = df['labels'].value_counts().idxmax()
df_majority = test[['text', 'labels']]
predictions = [majority_class] * len(df_majority)
df_majority['predictions'] = predictions
evaluation(df_majority, freq_0, freq_1)

"""# 3.	Classification by fine-tuning BERT (2.5 points)"""

# Preparing train data
train_df = train.iloc[:10000,:][['text','labels']] #+/- 80% of the training set
# Preparing eval data
eval_df = train.iloc[10000:,:][['text','labels']] #+/- 20% of the training set
# Preparing test data
test_df = test[['text','labels']]
test_list = test_df['text'].values.tolist()

print(len(train_df))
# print(train_df.head())
# print(len(eval_df))
# print(eval_df.head())
# print(len(test_list))
# print(test_list[:2])

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = ClassificationArgs()
model_args.overwrite_output_dir = True

cuda_available = torch.cuda.is_available()
print(cuda_available)
model = ClassificationModel(
    "bert", "bert-base-cased", use_cuda=cuda_available, args=model_args
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(test_list)

# Attach predictions to test df for evaluation
test_df['predictions'] = predictions
print(test_df)

# Calculate evaluation metrics
evaluation(test_df, freq_0, freq_1)

print(sum(test_df['predictions'] == test_df['labels']))
print(len(test_df['predictions']))

# Confusion matrix elements
print('TP: ', sum(test_df['TP']))
print('FN: ', sum(test_df['FN']))
print('FP: ', sum(test_df['FP']))
print('TN: ', sum(test_df['TN']))

"""# 4.	Inspect the tokenization of the OLIDv1 training set using the BERT???s tokenizer (2.5 points)"""

train_text = train['text'].values.tolist()
#print(train_text)
train_tokens = []
for i in range(len(train_text)):
  tokens = model.tokenizer.tokenize(train_text[i])
  train_tokens.extend(tokens)

# number of tokens
print(len(train_tokens))

# number of token split into subwords
train_tokens_str = ' '.join(train_tokens)
print(train_tokens_str.count('##'))
train_tokens_str[:10000]

# How long (in characters) is the longest subword in the BERT???s vocabulary? (0.5 points)
print(max(list(model.tokenizer.vocab.keys()), key=len))
print(len(max(list(model.tokenizer.vocab.keys()), key=len)))

"""# 5.	Typos (6 points) """

# reseed
np.random.seed(42)

# text
# subset_test = pd.read_csv('../input/olid-subset-diagnostic-tests.csv')
subset_test = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/olid-subset-diagnostic-tests.csv')
text = subset_test['text'].tolist()

# overload existing method from checklist to add more than 1 typo
def add_typos(string, typos=5):
        """Perturbation functions, swaps random characters with their neighbors
        Parameters
        ----------
        string : str
            input string
        typos : int
            number of typos to add
        Returns
        -------
        list(string)
            perturbed strings
        """
        string = list(string)
        swaps = np.random.choice(len(string) - 1, typos)
        for swap in swaps:
            tmp = string[swap]
            string[swap] = string[swap + 1]
            string[swap + 1] = tmp
        return ''.join(string)

# add_typos
typos = Perturb.perturb(text, add_typos)
#print(typos.data)

# plug back into pandas
new_text = []
for t in typos.data:
  new_text.append(t[1])
subset_test['text_typos'] = new_text
print(subset_test)

# using model trained previously
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

### Perturbed type test data ----------------------------------------------------

# Preparing test data
subset_test_df = subset_test[['text','labels']]
subset_test_list = subset_test_df['text'].values.tolist()

# Make predictions with the model on non-perturbed set
subset_predictions, subset_raw_outputs = model.predict(subset_test_list)
#print(predictions, raw_outputs)

# Attach predictions to non-perturbed df for evaluation
subset_test_df['predictions'] = subset_predictions


### Perturbed type test data ----------------------------------------------------
# Preparing perturbed typo test data
subset_typos_test_df = subset_test[['text_typos','labels']]
subset_typos_test_list = subset_typos_test_df['text_typos'].values.tolist()

# Make predictions with the model on perturbed set
subset_typos_predictions, subset_typos_raw_outputs = model.predict(subset_typos_test_list)
#print(predictions, raw_outputs)

# Attach predictions to typo perturbed df for evaluation
subset_typos_test_df['predictions'] = subset_typos_predictions


# get class distributions for getting weighted F1 score later --------------------
subset_freq_0 = subset_test_df['labels'].value_counts(normalize=True).iloc[0]
subset_freq_1 = subset_test_df['labels'].value_counts(normalize=True).iloc[1]
# note that class distributions are equal in this subset so weighted and macro
# F1 scores will be the same

# Calculate evaluation metrics for non-perturbed data
print('NON-PERTURBED DATASET:')
evaluation(subset_test_df, subset_freq_0, subset_freq_1)
print('Correctly identified messages in data: ' + str(sum(subset_test_df['predictions'] == subset_test_df['labels'])))
print('Total number of messages: ' + str(len(subset_test_df['predictions'])))
print('\n')

# Calculate evaluation metrics for perturbed data
print('PERTURBED TYPO DATASET:')
evaluation(subset_typos_test_df, subset_freq_0, subset_freq_1)
print('Correctly identified messages in typo data: ' + str(sum(subset_typos_test_df['predictions'] == subset_typos_test_df['labels'])))
print('Total number of messages: ' + str(len(subset_typos_test_df['predictions'])))
print('\n')

# save dataframes to html
subset_test_df.to_html('/content/gdrive/MyDrive/Colab Notebooks/subset_test_df.html')
subset_typos_test_df.to_html('/content/gdrive/MyDrive/Colab Notebooks/subset_typos_test_df.html')

# read dataframes from html
#subset_test_df = pd.read_html('../input/olid-subset-results/subset_test_df.html', index_col=0)[0]
#subset_typos_test_df = pd.read_html('../input/olid-subset-results/subset_typos_test_df.html', index_col=0)[0]

# The examples below are mislabeled as not containing 
# hate-speech by the model trained on the perturbed dataset:

print(subset_test_df['text'][16])
print(subset_typos_test_df['text_typos'][16])
print('\n')

print(subset_test_df['text'][21])
print(subset_typos_test_df['text_typos'][21])
print('\n')

print(subset_test_df['text'][28])
print(subset_typos_test_df['text_typos'][28])
print('\n')

"""# 6.	Negation (4.5 points) """

nlp = spacy.load('en_core_web_sm')
pdata = list(nlp.pipe(text))

# reseed
np.random.seed(42)

# text
# subset_test = pd.read_csv('../input/olid-subset-diagnostic-tests.csv')
subset_test = pd.read_csv('/content/gdrive/MyDrive/Colab Notebooks/olid-subset-diagnostic-tests.csv')
text = subset_test['text'].tolist()

# add_negations
negations = Perturb.perturb(pdata, Perturb.add_negation)
#print(negations.data)
#print(len(subset_test))
#print(len(negations.data))

ids = []
text = []
text_negations = []
labels = []

# build negations dataframe from scratch since the method uses spacy
for n in negations.data:
    for index, row in subset_test.iterrows():
        if n[0] == row['text']:
            ids.append(row['id'])
            text.append(n[0])            
            labels.append(row['labels'])
            text_negations.append(n[1])

# initialize data of lists.
neg_data = {'id': ids, 'text': text,  'labels': labels, 'text_negations': text_negations}
subset_test = pd.DataFrame(neg_data)
# print(subset_test)

# using model trained previously
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

### Perturbed type test data ----------------------------------------------------

# Preparing test data
subset_test_df = subset_test[['text','labels']]
subset_test_list = subset_test_df['text'].values.tolist()

# Make predictions with the model
subset_predictions, subset_raw_outputs = model.predict(subset_test_list)
#print(predictions, raw_outputs)

# Attach predictions to test df for evaluation
subset_test_df['predictions'] = subset_predictions


### Perturbed type test data ----------------------------------------------------

# Preparing negations test data
subset_negs_test_df = subset_test[['text_negations','labels']]
subset_negs_test_list = subset_negs_test_df['text_negations'].values.tolist()

# Make predictions with the model on negations dataset
subset_negs_predictions, subset_negs_raw_outputs = model.predict(subset_negs_test_list)
#print(predictions, raw_outputs)

# Attach predictions to negations test df for evaluation
subset_negs_test_df['predictions'] = subset_negs_predictions

# get class distributions for getting weighted F1 score later
subset_freq_0 = subset_test_df['labels'].value_counts(normalize=True).iloc[0]
subset_freq_1 = subset_test_df['labels'].value_counts(normalize=True).iloc[1]
# note that class distributions are equal in this subset so weighted and macro
# F1 scores will be the same

# Calculate evaluation metrics for non-perturbed data
print('NON-PERTURBED DATASET:')
evaluation(subset_test_df, subset_freq_0, subset_freq_1)
print('Correctly identified messages in data: ' + str(sum(subset_test_df['predictions'] == subset_test_df['labels'])))
print('Total number of messages: ' + str(len(subset_test_df['predictions'])))
print('\n')

# Calculate evaluation metrics for perturbed data
print('PERTURBED TYPO DATASET:')
evaluation(subset_negs_test_df, subset_freq_0, subset_freq_1)
print('Correctly identified messages in negations data: ' + str(sum(subset_negs_test_df['predictions'] == subset_negs_test_df['labels'])))
print('Total number of messages: ' + str(len(subset_negs_test_df['predictions'])))
print('\n')

# The examples below are mislabeled as not containing 
# hate-speech by the model trained on the perturbed dataset:

print(subset_test_df['text'][30])
print(subset_negs_test_df['text_negations'][30])
print('\n')

print(subset_test_df['text'][31])
print(subset_negs_test_df['text_negations'][31])
print('\n')

print(subset_test_df['text'][71])
print(subset_negs_test_df['text_negations'][71])
print('\n')

# save dataframes to html
#subset_test_df.to_html('subset_test_df.html')
subset_negs_test_df.to_html('/content/gdrive/MyDrive/Colab Notebooks/subset_negs_test_df.html')

# read dataframes from html
#subset_test_df = pd.read_html('../input/olid-subset-results/subset_test_df.html', index_col=0)[0]
#subset_typos_test_df = pd.read_html('../input/olid-subset-results/subset_typos_test_df.html', index_col=0)[0]

"""# 7. Creating examples from scratch with checklist  (2.5 points)"""

from checklist.editor import Editor
editor = Editor()

# getting masked language model suggestions
hate = editor.suggest('I hate {mask}.')[:30]
not_hate = editor.suggest('I don\'t hate {mask}.')[:30]

hate_mask = []
for i in hate:
    hate_mask.append('I hate ' + i)
#print(hate_mask)

not_hate_mask = []
for i in hate:
    not_hate_mask.append('I don\'t hate ' + i)
#print(not_hate_mask)

# using the built-in lexicon, we also explore more specific mask subjects
ret1 = editor.template('I hate {nationality}.')
ret2 = editor.template('I don\'t hate {nationality}.')
ret3 = editor.template('I hate {religion}.')
ret4 = editor.template('I don\'t hate {religion}.')

hate_nationalities = list(np.random.choice(ret1.data, 10))
nationalities = list(np.random.choice(ret2.data, 10))
hate_religions = list(np.random.choice(ret3.data, 10))
religions = list(np.random.choice(ret4.data, 10))

# build dataset
hate_list = hate_mask + not_hate_mask + hate_nationalities + nationalities + hate_religions + religions
hate_df = pd.DataFrame({'text': hate_list})
print(hate_df)

# Make predictions with the model on hate dataset
hate_predictions, hate_raw_outputs = model.predict(hate_list)

# Attach predictions to test df for evaluation
hate_df['predictions'] = hate_predictions

print(hate_df)

# save hate result to html
hate_df.to_html('/content/gdrive/MyDrive/Colab Notebooks/hate_df.html')

