# smote_balance.py
from imblearn.over_sampling import SMOTE
import numpy as np

def apply_smote(features, labels, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(features, labels)
    return X_res, y_res
