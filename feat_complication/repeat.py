import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from utils.helper import split_data



df_augment = pd.read_excel('../data_curation/compli_augment.xlsx')
df_augment = df_augment[df_augment['Diagnosis']==1]
df_base = pd.read_excel('../data_curation/dataset_complications.xlsx')

train, val, test = split_data(df_base)

# print(f'train : {train['Severity'].value_counts()}')
# print(f'val : {val['Severity'].value_counts()}')
# print(f'test : {test['Severity'].value_counts()}')

col = df_augment.columns
# print (col)

train = train[col]
print (train['Severity'].value_counts())

df_train = pd.concat([train, df_augment], ignore_index= True)
print (df_train['Severity'].value_counts())







