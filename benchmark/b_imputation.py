import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from sklearn.model_selection import cross_validate
from Pipeline.pipeline import Pipeline_Diagnosis
from utils.helper import split_data
from sklearn.metrics import make_scorer, recall_score, precision_score


strategy_cont = [None,'rf','linear']
strategy_others = [None, 'rf']

specificity = make_scorer(recall_score, pos_label=0)
npv = make_scorer(precision_score, pos_label=0)

scoring = {
    'roc_auc': 'roc_auc',
    'accuracy': 'accuracy',
    'specificity': specificity,
    'npv': npv
}

results = []

df=pd.read_excel('../data_curation/dataset_diagnosis.xlsx')

train_df, _ , _ = split_data(df)
x= train_df.drop(columns=['Diagnosis'])
y= train_df['Diagnosis']

for cont in strategy_cont:
    for others in strategy_others:
        pipeline = Pipeline_Diagnosis(strategy_cont=cont, strategy_others=others)
        result = cross_validate(pipeline, x, y, scoring=scoring, cv=10, error_score='raise', return_train_score=False)

        mean_scores = {metric: result[f'test_{metric}'].mean() for metric in scoring.keys()}
        std_scores = {f'{metric}_std': result[f'test_{metric}'].std() for metric in scoring.keys()}


        results.append({
            'strategy_cont': cont,
            'strategy_others': others,
            **mean_scores,
            **std_scores
        })

results_df = pd.DataFrame(results)
results_df.to_excel('b555_imputation.xlsx', index=False)