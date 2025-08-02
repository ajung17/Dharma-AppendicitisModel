from imputer import Selective_Imputer
import pandas as pd

df_780=pd.read_excel('dataset_780.xlsx')
df_430=pd.read_excel('dataset_430.xlsx')

df_780=df_780.dropna(subset=['Diagnosis'])

feat_all=['Nausea','Loss_of_Appetite','Peritonitis','WBC_Count','Neutrophil_Percentage','Body_Temperature','CRP','Ketones_in_Urine','Appendix_Diameter','Free_Fluids','Diagnosis']
feat_diag=[]
feat_iterative=['WBC_Count','Neutrophil_Percentage','CRP','Body_Temperature']
feat_knn=['Nausea','Loss_of_Appetite','Peritonitis','Free_Fluids']

df_780= df_780[feat_all]
df_430= df_430[feat_all]


print(df_780)


