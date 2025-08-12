import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from utils.helper import eval_summary, model_compare

# df_dharma=pd.read_excel('test1_Dharma.xlsx')
# df_xgb = pd.read_excel('test1_xgboost.xlsx')
# df_lgbm = pd.read_excel('test1_lgbm.xlsx')

df_dharma=pd.read_excel('test2_Dharma.xlsx')
df_xgb = pd.read_excel('test2_xgboost.xlsx')
df_lgbm = pd.read_excel('test2_lgbm.xlsx')

metrics = [
    'AUC_ROC',
    'Accuracy',
    'Sensitivity',
    'Specificity',
    'PPV',
    'NPV'   
]

dataframes = [('dharma', df_dharma), ('xgb', df_xgb), ('lgbm', df_lgbm)]

for name, df in dataframes:
    result = eval_summary(df, metrics)
    result.to_excel(f'metrics2_{name}.xlsx', index=False)

    if name != 'dharma':
        result_diff= model_compare(df_dharma,df, metrics)
        result_diff.to_excel(f'benchmark2_{name}.xlsx', index=False)





