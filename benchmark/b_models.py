import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from utils.helper import bootstrap_test, split_data
import joblib

# df=pd.read_excel('../data_curation/dataset_model.xlsx')
df=pd.read_excel('../data_curation/dataset_tools.xlsx')
col_model =['Nausea','Loss_of_Appetite','Peritonitis','Body_Temperature','WBC_Count','Neutrophil_Percentage','CRP','Ketones_in_Urine','Appendix_Diameter','Free_Fluids','Diagnosis']

# _, _ , test_df = split_data(df)
test_df = df[col_model]

x_test= test_df.drop(columns=['Diagnosis'])
y_test= test_df['Diagnosis']

Dharma = joblib.load('model_Dharma.joblib')
xgb = joblib.load('model_XGBoost.joblib')
lgbm = joblib.load('model_LightGBM.joblib')

models = [('Dharma', Dharma), ('xgboost', xgb), ('lgbm', lgbm)]

for name, model in models:
    results = bootstrap_test( x_test= x_test, y_test = y_test, model= model, n_bootstraps=5555)
    results.to_excel(f'test2_{name}.xlsx', index=False)


