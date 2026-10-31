import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_ipcw

HORIZON = 7.0 # Horizon set by the challenge

def ipcw_c_index(y_train, y_val, risk_scores) -> float:
    """
    Calculates the C-index
    :param risk_scores: the output of the model
    :param y_train: to estimate the distribution of the censoring
    :param y_val: validation labels
    :return: C-index at horizon HORIZON
    """
    return concordance_index_ipcw(y_train, y_val, risk_scores, tau=HORIZON)[0]

def get_cv(n_fold: int, seed: int=42) -> StratifiedKFold:
    """
    Create a StratifiedKFold CV
    :param n_fold: number of folds
    shuffle=True : the dataset may be sorted by center
    :return: StratifiedKFold
    """
    return StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

def run_cv_cox(model, X: pd.DataFrame, y: pd.DataFrame, cv: StratifiedKFold, scorer) -> tuple :
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
    for fold, (train_index, val_index) in enumerate(cv.split(X, y["OS_STATUS"])):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)

        scores.append({"fold": fold, "train_sore": scorer(y_train, y_train, model.predict(X_train)),
        "val_sore": scorer(y_train, y_val, model.predict(X_val))})
    return pd.DataFrame(scores), pd.DataFrame(scores)["val_sore"].mean()
