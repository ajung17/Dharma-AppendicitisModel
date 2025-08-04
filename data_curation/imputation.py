import sys
import os
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.imputer import Selective_Impute
import pandas as pd

df=pd.read_excel('dataset_unique.xlsx',sheet_name=0)
print(df.shape)

feat_all=['Coughing_Pain','Body_Temperature', 'WBC_Count', 'Neutrophil_Percentage', 'CRP', 'Nausea','Migratory_Pain','Peritonitis','Ipsilateral_Rebound_Tenderness',
           'Loss_of_Appetite', 'Ketones_in_Urine', 'Free_Fluids']
feat_inlammatory= ['Body_Temperature', 'WBC_Count', 'Neutrophil_Percentage', 'CRP']
feat_others= ['Coughing_Pain','Nausea','Migratory_Pain','Peritonitis','Ipsilateral_Rebound_Tenderness', 'Loss_of_Appetite', 'Ketones_in_Urine', 'Free_Fluids']

dataset_test= Selective_Impute(dataset=df, col_all=feat_all, col_iter=feat_inlammatory, col_knn=feat_others)
# dataset_test.to_excel('dataset_imputed.xlsx', index=False)



