import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

def multiclass_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1, "confusion_matrix": cm}

def binary_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None
    return {"accuracy": acc, "precision": p, "recall": r, "specificity": spec, "f1": f1, "auroc": auc,
            "tn_fp_fn_tp": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}
