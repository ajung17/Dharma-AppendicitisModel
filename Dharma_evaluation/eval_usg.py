import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.helper import model_compare, eval_summary

import pandas as pd

df_dharma = pd.read_excel('Dharma_usg.xlsx')
df_usg = pd.read_excel('../analyses/metric_usg_6.xlsx')

metrics = [
    'AUC_ROC',
    'Accuracy',
    'Sensitivity',
    'Specificity',
    'PPV',
    'NPV'   
]

summary = eval_summary(df=df_dharma,metrics=metrics)
summary.to_excel('dharma_usg_summary.xlsx', index = False)

compare = model_compare(df1=df_dharma, df2=df_usg, metrics=metrics)
compare.to_excel('dharma_vs_usg.xlsx', index= False)


