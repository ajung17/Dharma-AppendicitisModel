from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

class Models:
    def __init__(self):
        self.models = {
            'Dharma': RandomForestClassifier(
                n_estimators=555,
                min_samples_split=12,
                min_samples_leaf=1,
                max_depth=35,
                random_state=17,
                class_weight="balanced"),
            'XGBoost': xgb.XGBClassifier(random_state=17),
            'LightGBM': lgb.LGBMClassifier(random_state=17)
        }
        
    def get_model(self, model_name):
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} is not defined. Available models: {list(self.models.keys())}")
        
    