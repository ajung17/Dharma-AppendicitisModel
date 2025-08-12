import sys
import os
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.imputer import Dharma_Imputer
import pandas as pd
import numpy as np

df=pd.read_excel('dataset_tools.xlsx',sheet_name=0)
# print(df.shape)
# print(df.isna().sum())

drop_col = ['Diagnosis','Severity','Alvarado_Score','Paedriatic_Appendicitis_Score']
df1= df.drop(columns=drop_col)


feat_all = list(df1.columns)
feat_inlammatory= ['Body_Temperature', 'WBC_Count', 'Neutrophil_Percentage', 'CRP']
feat_others= ['Coughing_Pain','Nausea','Migratory_Pain','Lower_Right_Abd_Pain','Peritonitis','Ipsilateral_Rebound_Tenderness',
               'Loss_of_Appetite', 'Ketones_in_Urine', 'Free_Fluids']
feat_flag= ['Appendix_Diameter']

dataset_test= Dharma_Imputer(feat_continuous=feat_inlammatory, feat_categorical=feat_others, feat_model=feat_all, feat_flag=feat_flag)

imputed_data = dataset_test.fit_transform(df1[feat_all])
# print(imputed_data.shape)
# print(imputed_data.isna().sum())

imputed_data = imputed_data.replace(-1, np.nan)
# print(imputed_data.index.equals(df.index))

imputed_data = pd.concat([imputed_data,df['Diagnosis']], axis= 1)
imputed_data.to_excel('dataset_tools_imputed.xlsx', index=False)

