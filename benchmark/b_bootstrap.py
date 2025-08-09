import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from Pipeline.pipeline import Pipeline_Diagnosis
from utils.models import Models_Diagnosis
from utils.helper import split_data, bootstrap
from sklearn.metrics import make_scorer, recall_score, precision_score

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
model_xgboost = Models_Diagnosis()
model_lgbm= Models_Diagnosis()

XGBoost= model_xgboost.get_model(model_name='XGBoost')
LightGBM= model_lgbm.get_model(model_name='LightGBM')


models=[Dharma, XGBoost, LightGBM]

results={}

for model in models:
    key = getattr(model, 'model_name', type(model).__name__)
    results[key] = bootstrap(x_train=x, y_train=y, model=model, scoring=scoring)

df=pd.DataFrame(results)

df.to_excel('bootstrap_3_models.xlsx').T
















