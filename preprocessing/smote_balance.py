# smote_balance.py
# Feature-level SMOTE balancing (not applied directly on images)

import numpy as np
from imblearn.over_sampling import SMOTE


def apply_smote(features: np.ndarray, labels: np.ndarray, random_state: int = 42):
    """
    Apply SMOTE to balance feature embeddings.

    Args:
        features: Feature matrix of shape (N, D)
        labels: Class labels of shape (N,)
        random_state: Random seed

    Returns:
        Balanced features and labels
    """

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(features, labels)
    return X_resampled, y_resampled
