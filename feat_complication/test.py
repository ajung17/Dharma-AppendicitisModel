import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import joblib
from utils.helper import split_data,performance_metrics_ci
from sklearn.metrics import confusion_matrix

dharma_comp = joblib.load('dharma/dharma_comp.joblib')

dharma_imputer = joblib.load('../models/Dharma_imputer.joblib')

df_base = pd.read_excel('../data_curation/dataset_complications.xlsx')
feat_model = ['Nausea','Loss_of_Appetite','Peritonitis','Body_Temperature','WBC_Count','Neutrophil_Percentage','CRP','Ketones_in_Urine','Appendix_Diameter','Free_Fluids','Severity']


df_model = df_base[feat_model]
_,_,test = split_data(df_model)

x_test = test.drop(columns = 'Severity')
y_test = test['Severity']

x_test_imputed = dharma_imputer.transform(x_test)


y_prob = dharma_comp.predict_proba(x_test_imputed)[:,1]

results = performance_metrics_ci(y_prob,y_test,threshold=0.44)

flat_results = {
    "Metric": ["Sensitivity", "Specificity", "PPV", "NPV"],
    "Estimate": [
        results["Sensitivity"],
        results["Specificity"],
        results["PPV"],
        results["NPV"]
    ],
    "CI_Lower": [
        results["Sensitivity_CI"][0],
        results["Specificity_CI"][0],
        results["PPV_CI"][0],
        results["NPV_CI"][0]
    ],
    "CI_Upper": [
        results["Sensitivity_CI"][1],
        results["Specificity_CI"][1],
        results["PPV_CI"][1],
        results["NPV_CI"][1]
    ]
}

df_results = pd.DataFrame(flat_results)
df_results.to_excel('test_metrics.xlsx', index=False)

