import pandas as pd
import numpy as np
from sksurv.util import Surv

def join_clinical_target(X_clinical:pd.DataFrame, Y:pd.DataFrame) -> pd.DataFrame:
    """
    Joining the two dataframes into a single dataframe, one to one on 'ID'
    :param X_clinical: the clinical features dataset
    :param Y: the target dataset
    :return: the merge dataframe
    """
    X, y = X_clinical.copy(), Y.copy()

    merge = pd.merge(X, y, how='left', on='ID', validate='one_to_one')
    merge.set_index('ID', inplace=True)
    return merge

def drop_missing_target(merged:pd.DataFrame) -> tuple:
    """
    Drop the missing target
    :param merged: the merged dataframe
    :return: the dataset without missing target and the number of missing targets
    """
    out = merged.copy()
    out.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)
    n_dropped = len(merged) - len(out)
    return out, n_dropped

def load_clinical_processed(clinical_train_processed_path) -> pd.DataFrame:
    """
    Loading the CSV dataset as a pandas dataframe
    :param clinical_train_processed_path: path of the processed clinical train file
    :return: the features and labels datasets
    """
    clinical_dataset = pd.read_csv(clinical_train_processed_path, index_col=0)
    Y = clinical_dataset[['OS_YEARS', 'OS_STATUS']]
    X = clinical_dataset.drop(['OS_YEARS', 'OS_STATUS'], axis=1)
    return X, Y

def make_surv_target(os_years: pd.Series, os_status: pd.Series) -> np.ndarray:
    """
    Build the structured array for the survival target
    :param os_years: follow-up duration
    :param os_status: status of the patient at the end of the follow-up
    :return:
    """
    return Surv.from_arrays(event=os_status, time=os_years)
