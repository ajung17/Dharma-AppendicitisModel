import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from utils.helper import split_data, bootstrap_test, eval_summary, model_compare
import joblib

df1=pd.read_excel('../data_curation/dataset_model.xlsx')
df2=pd.read_excel('../data_curation/dataset_tools.xlsx')

diag_col = ['Nausea','Loss_of_Appetite','Peritonitis','Body_Temperature','WBC_Count','Neutrophil_Percentage','CRP','Ketones_in_Urine','Appendix_Diameter','Free_Fluids','Diagnosis']

df1 = df1[diag_col]
df2 = df2[diag_col]

_, _ , test_df = split_data(df1)

test_df = test_df[test_df['Appendix_Diameter'].isna()]
df2 = df2[df2['Appendix_Diameter'].isna()]

test_df = test_df[test_df['Appendix_Diameter'].isna()].reset_index(drop=True)
df2 = df2[df2['Appendix_Diameter'].isna()].reset_index(drop=True)

df_nousg = pd.concat([df2, test_df], ignore_index=True)
df_nousg.to_excel('../data_curation/no_usg.xlsx', index=False)

# x_test = df_nousg.drop(columns='Diagnosis')
# y_test = df_nousg['Diagnosis']

# Dharma = joblib.load('../benchmark/model_Dharma.joblib')

# threshold =0.44
# results = bootstrap_test( x_test = x_test, y_test = y_test, model = Dharma, threshold=threshold, n_bootstraps = 5555)   
# results.to_excel('dharma_nousg_thres.xlsx', index=False)

# metrics = [
#     'AUC_ROC',
#     'Accuracy',
#     'Sensitivity',
#     'Specificity',
#     'PPV',
#     'NPV'   
# ]

# summary = eval_summary(df=results, metrics=metrics)
# summary.to_excel(f'dharma_nousg_thres{threshold}_summary.xlsx', index = False)









