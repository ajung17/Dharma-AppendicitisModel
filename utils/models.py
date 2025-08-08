from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import copy

def set_random_state(params, default_state=17):
    params = copy.deepcopy(params)  
    if 'random_state' not in params:
        params['random_state'] = default_state
    return params

class Models_Diagnosis:
    def __init__(self, rf_params=None, xgb_params=None, lgbm_params=None):
        self.models = {
            'Dharma': RandomForestClassifier(
                **set_random_state(rf_params if rf_params else {
                    'n_estimators': 555,
                    'min_samples_split': 12,
                    'min_samples_leaf': 1,
                    'max_depth': 35,
                    'class_weight': "balanced"
                }),
            ),
            'XGBoost': xgb.XGBClassifier(
                **set_random_state(xgb_params if xgb_params else {}),
            ),
            'LightGBM': lgb.LGBMClassifier(
                **set_random_state(lgbm_params if lgbm_params else {}),
            )
        }
        
    def get_model(self, model_name):
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' is not defined. Available models: {list(self.models.keys())}")

class Models_Complications:
    def __init__(self, rf_params=None, xgb_params=None, lgbm_params=None):
        self.models = {
            'Dharma': RandomForestClassifier(
                **set_random_state(rf_params if rf_params else {
                    'n_estimators': 555,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_depth': 33
                }),
            ),
            'XGBoost': xgb.XGBClassifier(
                **set_random_state(xgb_params if xgb_params else {}),
            ),
            'LightGBM': lgb.LGBMClassifier(
                **set_random_state(lgbm_params if lgbm_params else {}),
            )
        }
        
    def get_model(self, model_name='Dharma'):
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' is not defined. Available models: {list(self.models.keys())}")
