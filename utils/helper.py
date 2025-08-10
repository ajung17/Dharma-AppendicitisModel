from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.base import clone
from scipy import stats
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s — %(levelname)s — %(message)s"
)

def split_data(df, random_state=None):
    if random_state is None:
        random_state = 88
    temp_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    train_df, val_df = train_test_split(temp_df, test_size=0.25, random_state=random_state)
    return train_df, val_df, test_df


def bootstrap( x_train, y_train, model=None, scoring=None,n_bootstraps = 10):   
    rng = np.random.RandomState(17)
    cv= StratifiedKFold(n_splits=10, shuffle=True, random_state=88)

    results=[]

    for i in range(n_bootstraps):
        logging.info(F'STARTING ON SAMPLE_NO : {i+1} ')
        boot_indices = rng.choice(len(y_train), size=len(y_train), replace=True)
        x_boot = x_train.iloc[boot_indices,:]
        y_boot = y_train.iloc[boot_indices]

        mean_scores = {}
        
        model0=clone(model)

        result = cross_validate(
        model0,
        x_boot,
        y_boot,
        scoring=scoring,
        cv=cv,
        return_train_score=False,
        )

        for metric in scoring.keys():
            scores = result[f'test_{metric}']
            mean_scores[metric] = np.mean(scores)

        results.append({
            'sample': f'bootstrap_{i+1}',
            **mean_scores,
        })

        logging.info(f'BOOTSTRAPPING ON SAMPLE {i+1} ENDED')

    df=pd.DataFrame(results)

    return df


def eval_summary(df, metrics):
    summary = []
    
    for metric in metrics:
        scores = df[f'{metric}']
        
        ci_lower = np.percentile(scores, 2.5)
        ci_upper = np.percentile(scores, 97.5)

        summary.append({
            "metric": metric,
            "mean": np.mean(scores),
            "std": np.std(scores, ddof=1),  
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })
    
    return pd.DataFrame(summary)




    
    







