import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from Pipeline.pipeline import Pipeline_Diagnosis
from utils.helper import split_data
from sklearn.metrics import make_scorer, recall_score, precision_score



# strategy = ['dt','knn','linear']
strategy = ['dt']

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

for strat in strategy:
    pipeline = Pipeline_Diagnosis(strategy=strat, feat_flag=feat_flag)
    result = cross_validate(pipeline, x, y, scoring=scoring, cv=cv, error_score='raise', return_train_score=False)

    mean_scores = {metric: result[f'test_{metric}'].mean() for metric in scoring.keys()}
    std_scores = {f'{metric}_std': result[f'test_{metric}'].std() for metric in scoring.keys()}


    results.append({
        'strategy': strat,
        **mean_scores,
        **std_scores
    })

results_df = pd.DataFrame(results)
results_df.to_excel('b18_imputation.xlsx', index=False)