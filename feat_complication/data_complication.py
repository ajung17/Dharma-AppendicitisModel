import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from utils.helper import split_data
from imblearn.under_sampling import NearMiss
import joblib
from sklearn.utils import shuffle


dharma_imputer = joblib.load('../models/Dharma_Imputer.joblib')


df_augment = pd.read_excel('../data_curation/compli_augment.xlsx')
df_augment = df_augment[df_augment['Diagnosis']==1]
df_base = pd.read_excel('../data_curation/dataset_complications.xlsx')

train, val, test = split_data(df_base)

col = df_augment.columns

train = train[col]
print (train['Severity'].value_counts())

df_train = pd.concat([train, df_augment], ignore_index= True)
print (df_train['Severity'].value_counts())

y_train = df_train['Severity']
x_train = df_train.drop(columns='Severity')

x_train = dharma_imputer.fit_transform(x_train)

print(f'y_train : {y_train.value_counts()}')

nm = NearMiss(version=1, n_neighbors=3, sampling_strategy=0.65)  

x_train_nm, y_train_nm = nm.fit_resample(x_train, y_train)
print (y_train_nm.value_counts())

x_train_nm = pd.DataFrame(x_train_nm, columns=x_train.columns)
y_train_nm = pd.Series(y_train_nm, name='Severity')

train_nm = pd.concat([x_train_nm, y_train_nm], axis=1)

train_nm = shuffle(train_nm, random_state=17).reset_index(drop=True)




train_nm.to_excel('data_train_imputed.xlsx', index=False)















