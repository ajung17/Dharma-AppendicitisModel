import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from feat_complication.models import Models_Complications
from utils.helper import bootstrap,eval_summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer

data = pd.read_excel('../data_train_imputed.xlsx')
x_train = data.drop(columns= 'Severity')
y_train = data['Severity']

specificity = make_scorer(recall_score, pos_label=0)
npv = make_scorer(precision_score, pos_label=0)

scoring = {
    'AUC_ROC': 'roc_auc',
    'Accuracy': 'accuracy',
    'Specificity': specificity,
    'NPV': npv,
    'Sensitivity': 'recall',
    'PPV': 'precision'
    
}

metrics = [
    'AUC_ROC',
    'Accuracy',
    'Sensitivity',
    'Specificity',
    'PPV',
    'NPV'   
]

models = Models_Complications()

rf = models.get_model(model_name='rf')
xgb_simple = models.get_model(model_name='xgb_simple')
xgb_complex = models.get_model(model_name='xgb_complex')

models = [('xgb_simple', xgb_simple), ('xgb_complex', xgb_complex),('rf', rf)]
for name, model in models:
    results = bootstrap(x_train, y_train, model = model, scoring = scoring)
    results.to_excel(f'bootstrap_cv_{name}.xlsx', index=False)
    summary = eval_summary(results, metrics)
    summary.to_excel(f'summary_{name}.xlsx', index=False)

