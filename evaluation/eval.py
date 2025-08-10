import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from utils.helper import eval_summary

df=pd.read_excel('bootstrap_metrics.xlsx')

metrics = {
    'roc_auc',
    'accuracy',
    'specificity',
    'npv',
    'sensitivity',
    'ppv'   
}

dharma_results= eval_summary(df, metrics)

dharma_results.to_excel('eval_Dharma.xlsx', index=False)





