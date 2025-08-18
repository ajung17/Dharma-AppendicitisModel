import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from utils.helper import model_compare

df_dharma = pd.read_excel('cross-validation/bootstrap_cv_rf.xlsx')
df_xgb_simple = pd.read_excel('cross-validation/bootstrap_cv_xgb_simple.xlsx')
df_xgb_complex = pd.read_excel('cross-validation/bootstrap_cv_xgb_complex.xlsx')

metrics = [
    'AUC_ROC',
    'Accuracy',
    'Sensitivity',
    'Specificity',
    'PPV',
    'NPV'   
]

dataframes = [('xgb_simple', df_xgb_simple), ('xgb_complex', df_xgb_complex)]

for name, df in dataframes:
    result_diff= model_compare(df_dharma,df, metrics)
    result_diff.to_excel(f'models_compare/benchmark2_{name}.xlsx', index=False)




