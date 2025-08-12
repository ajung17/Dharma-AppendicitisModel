from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
import numpy as np
from sklearn.base import clone
import pandas as pd
import logging
from scipy import stats
from sklearn.metrics import  recall_score, precision_score, roc_auc_score, accuracy_score


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

def model_compare(df1,df2, metrics):
    summary = []

    for metric in metrics:
        scores1 = df1[f'{metric}']
        scores2 = df2[f'{metric}']
        diff = scores1 - scores2
        
        ci_lower = np.percentile(diff, 2.5)
        ci_upper = np.percentile(diff, 97.5)

        t_stat, p_ttest = stats.ttest_rel(scores1, scores2)

        if p_ttest < 1e-6:
            p_val_display = "<1e-6"
        else:
            p_val_display = f"{p_ttest:.6e}"

        summary.append({
            "metric": metric,
            "mean_diff": np.mean(diff),
            "std_diff": np.std(diff, ddof=1),  
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_ttest": p_val_display,
            "t_stat": t_stat

        })

    return pd.DataFrame(summary)

def bootstrap_test( x_test, y_test, model=None, n_bootstraps = 5555):   
    rng = np.random.RandomState(17)

    results=[]

    for i in range(n_bootstraps):
        logging.info(F'STARTING ON SAMPLE_NO : {i+1} ')
        boot_indices = rng.choice(len(y_test), size=len(y_test), replace=True)
        x_boot = x_test.iloc[boot_indices,:]
        y_boot = y_test.iloc[boot_indices]

        y_pred = model.predict(x_boot)
        y_pred_proba = model.predict_proba(x_boot)[:, 1]

        auroc_score = roc_auc_score(y_boot, y_pred_proba)
        ppv_score = precision_score(y_boot, y_pred, pos_label=1)
        npv_score = precision_score(y_boot, y_pred, pos_label=0)
        sensitivity_score = recall_score(y_boot, y_pred, pos_label=1)
        specificity_score = recall_score(y_boot, y_pred, pos_label=0)
        accuracy = accuracy_score(y_boot, y_pred)

        results.append({
            'sample': f'bootstrap_{i+1}',
            'AUC_ROC': auroc_score,
            'Accuracy': accuracy,
            'Sensitivity': sensitivity_score,
            'Specificity': specificity_score,
            'PPV': ppv_score,
            'NPV': npv_score
        })

        logging.info(f'BOOTSTRAPPING ON SAMPLE {i+1} ENDED')

    df=pd.DataFrame(results)

    return df

    

    






    
    







