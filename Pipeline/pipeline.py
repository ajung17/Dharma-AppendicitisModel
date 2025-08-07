import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.pipeline import Pipeline
from utils.imputer import Dharma_Imputer
from utils.models import Models_Diagnosis

def Pipeline_Diagnosis(model_name='Dharma', rf_params=None, xgb_params=None, lgbm_params=None,strategy_cont=None, strategy_others=None, reason=None):
    models_diag = Models_Diagnosis(rf_params, xgb_params, lgbm_params)
    model = models_diag.get_model(model_name)

    pipeline = Pipeline(steps=[
        ('imputer', Dharma_Imputer(strategy_cont=strategy_cont, strategy_others=strategy_others, reason=reason)),
        ('model', model)
    ])
    return pipeline


