
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from Pipeline.pipeline import Pipeline_Diagnosis
from utils.helper import split_data, bootstrap
from sklearn.metrics import make_scorer, recall_score, precision_score
from utils.models import Models_Diagnosis

import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s — %(levelname)s — %(message)s" )

df=pd.read_excel('../data_curation/dataset_model.xlsx')

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

_, _ , test_df = split_data(df)

x= test_df.drop(columns=['Diagnosis'])
y= test_df['Diagnosis']

feat_flag = ['Appendix_Diameter']

Dharma = Pipeline_Diagnosis(feat_flag=feat_flag)

models = Models_Diagnosis()

xgboost = models.get_model(model_name='XGBoost')
lgbm = models.get_model(model_name='LightGBM')

# models_list = [Dharma, xgboost]

# i=0
# for model in models_list:
#     i=i+1
#     logging.info(f'starting evaluation for model {i}')

#     result = bootstrap( x_train=x, y_train=y, model=model, scoring=scoring,n_bootstraps = 555) 
#     result.to_excel(f'b_test_{i}.xlsx', index=False)

#     logging.info(f'evaluation completed for model {i}')


i=3

result = bootstrap( x_train=x, y_train=y, model=lgbm, scoring=scoring,n_bootstraps = 555) 
result.to_excel(f'b_test_{i}_lgbm.xlsx', index=False)








