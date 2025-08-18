import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

data = pd.read_excel('../data_train_imputed.xlsx')

rf = RandomForestClassifier(random_state=17)

params_grid_rf = {
    "n_estimators": [ 333, 555 , 777, 888],       
    "max_depth": [3, 7, 10],              
    "min_samples_split": [2, 5, 10],       
    "min_samples_leaf": [1, 3, 5],         
    "max_features": ["sqrt", "log2"],      
    "class_weight": [ "balanced", {0:1, 1:2}, {0:1, 1:3},  {0:1, 1:5}]  
}

scoring = {
    'Recall': 'recall',
    'AUC': 'roc_auc',
    'AUPR': 'average_precision'
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=17)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=params_grid_rf,
    scoring=scoring, 
    refit='Recall',
    cv=cv,
    verbose=2,
    n_jobs=-1
)

x = data.drop(columns='Severity')
y = data['Severity']

grid_search.fit(x, y)

results = pd.DataFrame(grid_search.cv_results_)
results = results[[
    'rank_test_Recall', 
    'mean_test_Recall', 'std_test_Recall',
    'mean_test_AUC', 'std_test_AUC',
    'mean_test_AUPR', 'std_test_AUPR',
    'params'
]]

results_sorted = results.sort_values(by='mean_test_Recall', ascending=False)
results_sorted.to_excel('rf_params.xlsx', index=False)
