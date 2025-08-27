import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from Pipeline.pipeline import Pipeline_Diagnosis
from utils.helper import split_data, bootstrap
from sklearn.metrics import make_scorer, recall_score, precision_score


import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s — %(levelname)s — %(message)s" )

specificity = make_scorer(recall_score, pos_label=0)
npv = make_scorer(precision_score, pos_label=0)

scoring = {
    'roc_auc': 'roc_auc',
    'accuracy': 'accuracy',
    'specificity': specificity,
    'npv': npv,
    'sensitivity': 'recall',
    'ppv': 'precision'
    
}

df=pd.read_excel('../data_curation/dataset_model.xlsx')

train_df, _ , _ = split_data(df)



x= train_df.drop(columns=['Diagnosis'])
y= train_df['Diagnosis']

feat_flag = ['Appendix_Diameter']

Dharma = Pipeline_Diagnosis(feat_flag=feat_flag)

results= bootstrap( x_train=x, y_train=y, model=Dharma, scoring=scoring, n_bootstraps = 555)

results.to_excel('bootstrap_metrics.xlsx', index=False)


