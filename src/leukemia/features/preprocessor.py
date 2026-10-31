import numpy as np
import pandas as pd

HEMO_PARAMETERS = ['BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'HB', 'PLT'] # Blood parameters
HEMO_PARAMETERS_LOG = ['BM_BLAST', 'WBC', 'ANC', 'MONOCYTES', 'PLT'] # Columns we are transforming

def add_missing_indicator(clinical_dataset: pd.DataFrame, hemo_parameters=HEMO_PARAMETERS) -> pd.DataFrame:
    """
    Add missing value indicators for the blood parameters
    :param clinical_dataset: the clinical dataset we are transforming
    :param hemo_parameters: the list of blood parameters
    :return: the clinical dataset with missing indicators
    """
    out = clinical_dataset.copy()
    for var in hemo_parameters:
        out[var + '_MISSING'] = out[var].isna().astype(int)
    return out

def log_transform(clinical_dataset: pd.DataFrame, hemo_parameters_log=HEMO_PARAMETERS_LOG) -> pd.DataFrame:
    """
    Log1p transforming of the right-skewed blood parameters
    :param clinical_dataset: the clinical dataset we are transforming
    :param hemo_parameters_log: the list of the right-skewed blood parameters
    :return: the transformed clinical dataset
    """
    out = clinical_dataset.copy()
    for var in hemo_parameters_log:
        out[var] = np.log1p(out[var])
    out.rename(columns={v: f'{v}_LOG' for v in hemo_parameters_log}, inplace=True)
    return out

def add_cytogenetics_flags(clinical_dataset:pd.DataFrame) -> pd.DataFrame:
    """
    Add cytogenetics flags: Normal, Abnormal, Complex and Missing. A Complex karyotype is also Abnormal.
    :param clinical_dataset: the clinical dataset we are transforming
    :return: the transformed clinical dataset with cytogenetics flags
    """
    out = clinical_dataset.copy()
    cyto = out['CYTOGENETICS']
    # Creating masks
    is_missing = cyto.isna()
    is_normal = cyto.str.match(r'46,(xx|xy)(\[\d+\])?$', case=False, na=False)
    is_complex = cyto.str.match('complex', case=False, na=False)

    # Creating the normal feature
    out['CYTO_NORMAL'] = (is_normal & ~is_missing).astype(int)
    # Creating the complex feature
    out['CYTO_COMPLEX'] = (is_complex & ~is_missing).astype(int)
    # Creating the abnormal feature
    out['CYTO_ABNORMAL'] = (~is_normal & ~is_missing).astype(int)
    # Creating the missing feature
    out['CYTO_MISSING'] = (is_missing).astype(int)

    out.drop(columns= 'CYTOGENETICS', inplace=True)

    return out

def preprocess_clinical_v1(clinical_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing the clinical dataset
    :param clinical_dataset: the clinical dataset we are transforming
    :return: the preprocessed clinical dataset
    """
    preprocessed_clinical = clinical_dataset.copy()
    preprocessed_clinical = add_missing_indicator(preprocessed_clinical)
    preprocessed_clinical = log_transform(preprocessed_clinical)
    preprocessed_clinical = add_cytogenetics_flags(preprocessed_clinical)
    return preprocessed_clinical
