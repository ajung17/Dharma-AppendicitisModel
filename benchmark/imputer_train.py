import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.imputer import Dharma_Imputer
import pandas as pd
from utils.helper import split_data
import joblib

df = pd.read_excel('../data_curation/dataset_model.xlsx')

train, val, test = split_data(df)

feat_all = ['Nausea','Loss_of_Appetite','Peritonitis','Body_Temperature','WBC_Count','Neutrophil_Percentage','CRP','Ketones_in_Urine','Appendix_Diameter','Free_Fluids','Severity']
feat_model = ['Nausea','Loss_of_Appetite','Peritonitis','Body_Temperature','WBC_Count','Neutrophil_Percentage','CRP','Ketones_in_Urine','Appendix_Diameter','Free_Fluids']
feat_flag = ['Appendix_Diameter']


df_impute = pd.concat([train,val], ignore_index=True)

dharma_imputer = Dharma_Imputer(feat_flag=feat_flag, feat_model=feat_model)

x_train = df_impute[feat_model]

dharma_imputer.fit(x_train)

joblib.dump(dharma_imputer, '../models/Dharma_Imputer.joblib')





