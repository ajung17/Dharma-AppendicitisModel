import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from utils.helper import split_data, tune_threshold
import joblib
from utils.imputer import Dharma_Imputer



df = pd.read_excel('../data_curation/dataset_complications.xlsx')

train_df, val_df, _ = split_data (df=df)

dharma_comp = joblib.load('../feat_complication/Dharma_comp.joblib')
df_imputer=pd.read_excel('../data_curation/dataset_model.xlsx')


feat_imp = ['Nausea','Loss_of_Appetite', 'Neutrophil_Percentage','Peritonitis','Ketones_in_Urine','Free_Fluids','CRP','WBC_Count','Body_Temperature','Appendix_Diameter']
feat_flag= ['Appendix_Diameter']

feat_model = ['Nausea','Loss_of_Appetite','Peritonitis','Ketones_in_Urine','Free_Fluids','CRP','WBC_Count','Body_Temperature','Appendix_Diameter']

x_val = val_df[feat_imp]
y_val = val_df['Severity']


imputer = Dharma_Imputer(feat_continuous=None, feat_categorical=None, feat_model=feat_model, feat_flag=feat_flag, placeholder= -1, strategy=None)



df_imputer = df_imputer[feat_imp]
train_imp, val_imp , _  = split_data(df_imputer)
train_full_df = pd.concat([train_imp, val_imp], axis=0, ignore_index=True)


imputer = Dharma_Imputer(feat_flag=feat_flag,feat_model=feat_model)
imputer.fit(train_full_df)

x_imputed = imputer.transform(x_val)

y_prob = dharma_comp.predict_proba(x_imputed)[:, 1] 

threshold = tune_threshold(y_true= y_val, y_prob=y_prob, mode='sensitivity', min_other=0.60)


print(threshold)


