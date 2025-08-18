
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import copy

def set_random_state(params, default_state=17):
    params = copy.deepcopy(params)  
    if 'random_state' not in params:
        params['random_state'] = default_state
    return params

class Models_Complications:
    def __init__(self, rf_params=None, xgb1_params=None, xgb2_params=None):
        self.models = {
            'rf': RandomForestClassifier(
                **set_random_state(rf_params if rf_params else {
                    'n_estimators': 888,
                    'min_samples_leaf': 5,
                    'min_samples_split': 5, 
                    'max_features': 'sqrt', 
                    'max_depth': 3, 
                    'class_weight': {0: 1, 1: 5}
                }),
            ),
            'xgb_simple': xgb.XGBClassifier(
                **set_random_state(xgb1_params if xgb1_params else {
                    'n_estimators' : 111,
                    'max_depth' : 3,
                    'min_child_weight' : 6,
                    'learning_rate': 0.01,
                    'scale_pos_weight' : 7
                }),
            ),
            'xgb_complex': xgb.XGBClassifier(
                **set_random_state(xgb2_params if xgb2_params else {
                    'colsample_bytree': 0.6,
                    'gamma': 1,
                    'learning_rate': 0.01,
                    'max_depth': 3,
                    'min_child_weight': 3,
                    'n_estimators': 111,
                    'reg_alpha': 0.1,
                    'reg_lambda': 10,
                    'scale_pos_weight': 7,
                    'subsample': 0.8
                }),
            )
        }
        
    def get_model(self, model_name='rf'):
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' is not defined. Available models: {list(self.models.keys())}")