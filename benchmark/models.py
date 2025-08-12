
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from Pipeline.pipeline import Pipeline_Diagnosis
from utils.helper import split_data, bootstrap
from utils.models import Models_Diagnosis
import joblib

import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s — %(levelname)s — %(message)s" )

df=pd.read_excel('../data_curation/dataset_model.xlsx')



train_df, val_df , _  = split_data(df)
train_full_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

x_train= train_full_df.drop(columns=['Diagnosis'])
y_train= train_full_df['Diagnosis']


feat_flag = ['Appendix_Diameter']

Dharma = Pipeline_Diagnosis(feat_flag=feat_flag)

models = Models_Diagnosis()
xgboost = models.get_model(model_name='XGBoost')
lgbm = models.get_model(model_name='LightGBM')


Dharma.fit(x_train,y_train)
xgboost.fit(x_train,y_train)
lgbm.fit(x_train,y_train)

joblib.dump(Dharma, "model_Dharma.joblib")
joblib.dump(xgboost, "model_XGBoost.joblib")
joblib.dump(lgbm, "model_LightGBM.joblib")















