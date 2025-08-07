import sys
import os
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.imputer import Dharma_Imputer
import pandas as pd

df=pd.read_excel('dataset_unique.xlsx',sheet_name=0)
print(df.shape)

feat_all=['Coughing_Pain','Body_Temperature', 'WBC_Count', 'Neutrophil_Percentage', 'CRP', 'Nausea','Migratory_Pain','Peritonitis','Ipsilateral_Rebound_Tenderness',
           'Loss_of_Appetite', 'Ketones_in_Urine','Appendix_Diameter', 'Free_Fluids']
feat_inlammatory= ['Body_Temperature', 'WBC_Count', 'Neutrophil_Percentage', 'CRP']
feat_others= ['Coughing_Pain','Nausea','Migratory_Pain','Peritonitis','Ipsilateral_Rebound_Tenderness', 'Loss_of_Appetite', 'Ketones_in_Urine', 'Free_Fluids']
feat_flag= ['Appendix_Diameter']

dataset_test= Dharma_Imputer(feat_continuous=feat_inlammatory, feat_categorical=feat_others, feat_flag=feat_flag)

imputed_data = dataset_test.fit_transform(df[feat_all])
imputed_data.to_excel('dataset_imputed2.xlsx', index=False)

print(imputed_data.isna().sum())
