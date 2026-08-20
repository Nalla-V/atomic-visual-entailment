"""Classifier configurations compared during AVE-LS training."""

import json

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False

PARAM_KEYS = [
    "learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization",
    "early_stopping", "validation_fraction", "objective", "num_class",
    "eval_metric", "n_estimators", "max_depth", "subsample",
    "colsample_bytree", "reg_lambda", "reg_alpha", "min_child_weight",
    "gamma", "tree_method", "n_jobs", "random_state",
]


def build_classifier_configs(random_state):
    configs = {
        "hgb_default": HistGradientBoostingClassifier(
            learning_rate=0.10, max_iter=100, max_leaf_nodes=31,
            l2_regularization=0.0, early_stopping="auto",
            validation_fraction=0.10, random_state=random_state),
        "hgb_low_lr_l2": HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=300, max_leaf_nodes=31,
            l2_regularization=0.05, early_stopping=True,
            validation_fraction=0.10, random_state=random_state),
        "hgb_low_lr_stronger_l2": HistGradientBoostingClassifier(
            learning_rate=0.04, max_iter=500, max_leaf_nodes=31,
            l2_regularization=0.10, early_stopping=True,
            validation_fraction=0.10, random_state=random_state),
    }

    if HAS_XGBOOST:
        common = dict(objective="multi:softprob", num_class=3,
                      eval_metric="mlogloss", random_state=random_state,
                      n_jobs=-1, tree_method="hist")
        configs.update({
            "xgb_default": XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.30,
                subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
                reg_alpha=0.0, min_child_weight=1.0, gamma=0.0, **common),
            "xgb_shallow_low_lr": XGBClassifier(
                n_estimators=1100, max_depth=3, learning_rate=0.025,
                subsample=0.88, colsample_bytree=0.88, reg_lambda=3.5,
                reg_alpha=0.25, min_child_weight=1.5, gamma=0.02, **common),
            "xgb_medium_lr": XGBClassifier(
                n_estimators=700, max_depth=3, learning_rate=0.05,
                subsample=0.90, colsample_bytree=0.90, reg_lambda=2.0,
                reg_alpha=0.10, min_child_weight=1.0, gamma=0.0, **common),
            "xgb_deeper_regularized": XGBClassifier(
                n_estimators=600, max_depth=4, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=4.0,
                reg_alpha=0.25, min_child_weight=2.0, gamma=0.05, **common),
        })

    return configs


def classifier_family(config_name):
    if config_name.startswith("hgb"):
        return "HistGradientBoosting"
    if config_name.startswith("xgb"):
        return "XGBoost"
    return "Other"


def export_hyperparameter_table(classifier_configs, csv_path, tex_path):
    rows = []
    for name, model in classifier_configs.items():
        params = model.get_params()
        row = {"classifier_config": name,
               "classifier_family": classifier_family(name)}
        for k in PARAM_KEYS:
            if k in params:
                row[k] = params[k]
        row["params_json"] = json.dumps(
            {k: params[k] for k in PARAM_KEYS if k in params}, ensure_ascii=False)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    latex_df = df[["classifier_config", "classifier_family", "params_json"]].copy()
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_df.to_latex(index=False, escape=True))

    return df