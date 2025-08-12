import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from utils.ss import tools_metrics
from utils.helper import eval_summary

df = pd.read_excel('tools.xlsx')
df_usg = df[['Appendix_Diameter', 'Diagnosis']].dropna()

dataframes = [
    ('as', df['Alvarado_score'], df['Diagnosis'], [5, 7]),
    ('pas', df['PAS_score'], df['Diagnosis'], [4, 6]),
    ('air', df['AIR_score'], df['Diagnosis'], [5, 9]),
    ('tzanaki', df['Tzanaki_score'], df['Diagnosis'], [8]),
    ('usg', df_usg['Appendix_Diameter'], df_usg['Diagnosis'], [6])
]

metrics = [
    'AUC_ROC',
    'Accuracy',
    'Sensitivity',
    'Specificity',
    'PPV',
    'NPV'   
]



for name, system, target, thresholds in dataframes:
    for thres in thresholds:
        result = tools_metrics(x=system, y=target, threshold=thres, n_bootstraps=5555)
        result_summary = eval_summary(result, metrics)

        result.to_excel(f'metric_{name}_{thres}.xlsx', index=False)
        result_summary.to_excel(f'summary_{name}_{thres}.xlsx', index=False)
