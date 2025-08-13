import sys
import os
from os import name

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s — %(levelname)s — %(message)s"
)

from sklearn.pipeline import Pipeline
from utils.imputer import Dharma_Imputer
from utils.models import Models_Diagnosis, Models_Complications

def Pipeline_Diagnosis(strategy=None, model_name='Dharma',feat_flag=None):

    models_diag = Models_Diagnosis()
    model = models_diag.get_model(model_name)

    dharma_pipeline = Pipeline(steps=[
        ('imputer', Dharma_Imputer(strategy=strategy, feat_flag=feat_flag)),
        ('model', model)
    ])

    return dharma_pipeline

def Pipeline_Complications(strategy=None, model_name = 'Dharma', feat_flag = None):

    models_comp = Models_Complications()
    model = models_comp.get_model(model_name)

    dharma_pipeline = Pipeline(steps=[
        ('imputer', Dharma_Imputer(strategy=strategy, feat_flag=feat_flag)),
        ('model', model)
    ])

    return dharma_pipeline
    

