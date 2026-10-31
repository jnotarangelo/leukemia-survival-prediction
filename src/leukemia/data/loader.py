import pandas as pd
import numpy as np
from sksurv.util import Surv

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
