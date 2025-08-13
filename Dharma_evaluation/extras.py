import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import joblib
from utils.helper import bootstrap_test

test_usg = pd.read_excel('../analyses/tools.xlsx')
df_usg = test_usg.dropna(subset=['Appendix_Diameter'])

diag_col = ['Nausea','Loss_of_Appetite','Peritonitis','Body_Temperature','WBC_Count','Neutrophil_Percentage','CRP','Ketones_in_Urine','Appendix_Diameter','Free_Fluids','Diagnosis']
df_model = df_usg[diag_col]

x_test = df_model.drop(columns='Diagnosis')
y_test = df_model['Diagnosis']

Dharma = joblib.load('../benchmark/model_Dharma.joblib')

# results = bootstrap_test( x_test = x_test, y_test = y_test, model = Dharma, n_bootstraps = 5555)   
# results.to_excel('Dharma_usg.xlsx', index=False)









