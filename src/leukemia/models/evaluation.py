import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_ipcw
from leukemia.data import loader

HORIZON = 7.0 # Horizon set by the challenge

def ipcw_c_index(y_train, y_val, risk_scores, tau=HORIZON) -> float:
    """
    Calculates the C-index
    :param risk_scores: the output of the model
    :param y_train: to estimate the distribution of the censoring
    :param y_val: validation labels
    :return: C-index at horizon HORIZON
    """
    return concordance_index_ipcw(y_train, y_val, risk_scores, tau=tau)[0]

def get_cv(n_fold: int, seed: int=42) -> StratifiedKFold:
    """
    Create a StratifiedKFold CV
    :param n_fold: number of folds
    shuffle=True : the dataset may be sorted by center
    :return: StratifiedKFold
    """
    return StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

def run_cv_cox(pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame, cv: StratifiedKFold, scorer) -> tuple :
    """
    Evaluate the cox model
    :param model: model we want to evaluate
    :param X: feature dataset
    :param y: target dataset
    :param cv: cross validation split
    :param scorer: scoring function: IPCW C-index
    :return: the cv results (n°fold, train_sore, val_sore) and the mean score of the validation sets
    """
    scores = []
    for fold, (train_index, val_index) in enumerate(cv.split(X_train, y_train["OS_STATUS"])):
        X_train_cv, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        y_train_cv = loader.make_surv_target(y_train_cv['OS_YEARS'], y_train_cv['OS_STATUS'])
        y_val = loader.make_surv_target(y_val['OS_YEARS'], y_val['OS_STATUS'])

        pipeline.fit(X_train_cv, y_train_cv)

        scores.append({"fold": fold, "train_sore": scorer(y_train_cv, y_train_cv, pipeline.predict(X_train_cv)),
        "val_sore": scorer(y_train_cv, y_val, pipeline.predict(X_val))})
        score_mean = {'mean': pd.DataFrame(scores)["val_sore"].mean(), 'std': pd.DataFrame(scores)["val_sore"].std()}

    return pd.DataFrame(scores), pd.DataFrame([score_mean])
