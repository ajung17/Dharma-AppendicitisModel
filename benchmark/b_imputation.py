import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from Pipeline.pipeline import Pipeline_Diagnosis
from utils.helper import split_data
from sklearn.metrics import make_scorer, recall_score, precision_score
from scipy import stats

strategy = ['dt','knn','linear','simple']

cv= StratifiedKFold(n_splits=10, shuffle=True, random_state=88)

specificity = make_scorer(recall_score, pos_label=0)
npv = make_scorer(precision_score, pos_label=0)

scoring = {
    'roc_auc': 'roc_auc',
    'accuracy': 'accuracy',
    'specificity': specificity,
    'npv': npv,
    'sensitivity': 'recall',
    'ppv': 'precision'
    
}

results = []

df=pd.read_excel('../data_curation/dataset_model.xlsx')

train_df, _ , _ = split_data(df)
x= train_df.drop(columns=['Diagnosis'])
y= train_df['Diagnosis']
feat_flag = ['Appendix_Diameter']

baseline_scores=[]
baseline_name=strategy[0]

for strat in strategy:
    pipeline = Pipeline_Diagnosis(strategy=strat, feat_flag=feat_flag)
    result = cross_validate(pipeline, x, y, scoring=scoring, cv=cv, error_score='raise', return_train_score=False)

    mean_scores = {}
    std_scores = {}
    ci_scores = {}
    p_values={}

    for metric in scoring.keys():
        scores = result[f'test_{metric}']
        mean_scores[f'{metric}_mean'] = np.mean(scores)
        std_scores[f'{metric}_std'] = np.std(scores, ddof=1)
      
        n_folds = len(scores)
        se = std_scores[f'{metric}_std'] / np.sqrt(n_folds)
        t_val = stats.t.ppf(0.975, df=n_folds-1)  
        ci_lower = mean_scores[f'{metric}_mean'] - t_val * se
        ci_upper = mean_scores[f'{metric}_mean'] + t_val * se
        ci_scores[f'{metric}_ci'] = f"[{ci_lower:.3f}, {ci_upper:.3f}]"

    roc_auc_scores = result['test_roc_auc']

    if strat == baseline_name:
        baseline_scores = roc_auc_scores
        p_values['roc_auc_p'] = None
    else:
        t_stat, p_val = stats.ttest_rel(baseline_scores, roc_auc_scores)
        p_values['roc_auc_p'] = p_val

    results.append({
        'strategy': strat,
        **mean_scores,
        **std_scores,
        **ci_scores,
        **p_values
    })

results_df = pd.DataFrame(results)
results_df.to_excel('b_imputation.xlsx', index=False)