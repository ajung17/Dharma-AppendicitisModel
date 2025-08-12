import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from utils.helper import model_compare

df_as_5 = pd.read_excel('../analyses/metric_as_5.xlsx')
df_as_7 = pd.read_excel('../analyses/metric_as_7.xlsx')

df_pas_4 = pd.read_excel('../analyses/metric_pas_4.xlsx')
df_pas_6 = pd.read_excel('../analyses/metric_pas_6.xlsx')

df_air_5 = pd.read_excel('../analyses/metric_air_5.xlsx')
df_air_9 = pd.read_excel('../analyses/metric_air_9.xlsx')

df_tzanaki = pd.read_excel('../analyses/metric_tzanaki_8.xlsx')
# df_usg = pd.read_excel('../analyses/metric_usg_6.xlsx')

df_dharma = pd.read_excel('../benchmark/test2_Dharma.xlsx')

metrics = [
    'AUC_ROC',
    'Accuracy',
    'Sensitivity',
    'Specificity',
    'PPV',
    'NPV'   
]

dataframes = [('as_5', df_as_5), ('as_7', df_as_7),('pas_4', df_pas_4), ('pas_6', df_pas_6),
              ('air_5', df_air_5),('air_9',df_air_9),('tzanaki',df_tzanaki)]

for ss, df in dataframes:
    result_diff= model_compare(df_dharma,df, metrics)
    result_diff.to_excel(f'eval_{ss}.xlsx',index=False)



