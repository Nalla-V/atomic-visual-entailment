"""Post-training analyses for AVE-LS.

    python -m src.selection.analysis

Data efficiency, permutation importance, and the selector-objective ablation,
with the figures each one produces. Run src.selection.train first.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.selection import features as F

SEP = "=" * 140

# Thesis plot colours: blue and wine-red.
PLOT_BLUE = "#4F81BD"
PLOT_WINE = "#C0504D"
BAR_COLORS = [PLOT_BLUE, PLOT_WINE]

VARIANT_LABELS = {
    "baseline_only": "Full-\nhypothesis",
    "simple_atomic": "Simple\natomic",
    "structured_atomic": "Structured\natomic",
    "full_12_methods": "Full\n12 methods",
}

OBJECTIVE_LABELS = {
    "label_level": "Label-level\n(3 labels)",
    "candidate_level": "Candidate-level\n(12 outputs)",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

METADATA_FEATURES = {
    "num_atoms", "hypothesis_word_count", "total_atom_word_count",
    "atom_bucket_1", "atom_bucket_2", "atom_bucket_3", "atom_bucket_4plus",
}

GENERATED_POOL_EXACT = {
    "num_unique_pred_labels", "all_methods_agree", "has_disagreement",
    "majority_count", "majority_type_is_majority", "majority_type_is_tie",
}

SCORE_POOL_EXACT = {
    "num_unique_score_pred_labels", "all_score_preds_agree",
    "has_score_pred_disagreement", "score_majority_count",
    "score_majority_type_is_majority", "score_majority_type_is_tie",
}


# ============================================================
# HELPERS
# ============================================================

def save_plot(fig, png_path, pdf_path=None):
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if pdf_path:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def viridis_colors(n):
    cmap = plt.cm.viridis
    if n == 1:
        return [cmap(0.7)]
    return [cmap(x) for x in np.linspace(0.25, 0.9, n)]


def parse_size_list(size_text, train_pool_size):
    out = []
    for item in size_text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "full":
            out.append("full")
        else:
            n = int(item)
            if n <= train_pool_size:
                out.append(n)
    if "full" not in out:
        out.append("full")
    return out


def parse_seed_list(seed_text):
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def stratified_subset_indices(train_idx, y, size, seed):
    if size >= len(train_idx):
        return np.array(train_idx, dtype=int)
    y_ids = y.map(F.LABEL_TO_ID).values
    _, subset_idx = train_test_split(
        train_idx, test_size=size, train_size=None,
        stratify=y_ids[train_idx], random_state=seed)
    return np.array(sorted(subset_idx), dtype=int)


def fit_model_on_indices(classifier_template, X, y, indices):
    y_ids = y.map(F.LABEL_TO_ID).values
    model = clone(classifier_template)
    model.fit(X.iloc[indices], y_ids[indices])
    return model


def macro_f1_from_ids(y_true_ids, pred_ids):
    return float(f1_score(y_true_ids, pred_ids, average="macro",
                          labels=[0, 1, 2], zero_division=0))


def assign_feature_type_family(feature_name):
    """Mutually exclusive subgroups for reporting. Parse flags return None."""
    if "parse_ok" in feature_name:
        return None
    if feature_name in METADATA_FEATURES:
        return "Decomposition metadata"
    if feature_name.startswith("agree_") or feature_name.startswith("score_agree_"):
        return "Pairwise agreement"
    if feature_name in GENERATED_POOL_EXACT:
        return "Generated-label pool"
    if (feature_name.startswith("num_votes_")
            or feature_name.startswith("frac_votes_")
            or feature_name.startswith("majority_label_")):
        return "Generated-label pool"
    if feature_name in SCORE_POOL_EXACT:
        return "Score-based label pool"
    if (feature_name.startswith("num_score_votes_")
            or feature_name.startswith("frac_score_votes_")
            or feature_name.startswith("score_majority_label_")):
        return "Score-based label pool"
    return "Per-method prediction features"


def display_feature_name(feature_name):
    """Shorten raw column names for plots. The model uses the raw columns."""
    name = feature_name
    name = name.replace("_baseline_", "_fh_")
    name = name.replace("_joint_", "_atomic_")
    name = name.replace("_self_decompose_", "_sd_")
    name = name.replace("_confidence", "_highest_score")
    name = name.replace("_margin", "_score_margin")
    name = name.replace("_normalized_entropy", "_normalized_score_entropy")
    name = name.replace("_entropy", "_score_entropy")
    return name


def prepare_classifier_for_num_classes(classifier_template, num_classes):
    """Adapt XGBoost's explicit num_class when the target space changes from
    3 labels to 12 candidate outputs."""
    model = clone(classifier_template)
    if "num_class" in model.get_params():
        model.set_params(num_class=int(num_classes))
    return model


def choose_oracle_candidate_winner(method_outputs, method_keys, gold):
    """If any candidate generates the gold label, choose the correct one with
    the highest margin, then confidence, then gold score. Otherwise choose the
    candidate assigning the highest likelihood to the gold label."""
    gold = F.normalize_label(gold)

    def correct_sort_key(method_key):
        out = method_outputs[method_key]
        scores = F.scores_to_probs(out.get("scores", {}))
        diag = F.confidence_diag(scores)
        return (float(out.get("margin", diag["margin"])),
                float(out.get("confidence_score", diag["confidence_score"])),
                float(scores.get(gold, 0.0)),
                -method_keys.index(method_key))

    correct = [mk for mk in method_keys
               if F.normalize_label(method_outputs[mk].get("prediction", "neutral")) == gold]
    if correct:
        return max(correct, key=correct_sort_key), "correct_highest_margin"

    def fallback_sort_key(method_key):
        out = method_outputs[method_key]
        scores = F.scores_to_probs(out.get("scores", {}))
        diag = F.confidence_diag(scores)
        return (float(scores.get(gold, 0.0)),
                float(out.get("margin", diag["margin"])),
                float(out.get("confidence_score", diag["confidence_score"])),
                -method_keys.index(method_key))

    return max(method_keys, key=fallback_sort_key), "no_correct_highest_gold_score"


# ============================================================
# FEATURE VARIANT COMPARISON
# ============================================================

def plot_feature_variant_comparison(results_df, paths):
    """For each feature variant, the best HGB and best XGB config, over
    validation macro-F1 and accuracy."""
    rows = []
    for variant in F.VARIANT_DISPLAY_ORDER:
        for family in ["HistGradientBoosting", "XGBoost"]:
            sub = results_df[(results_df["feature_variant"] == variant)
                             & (results_df["classifier_family"] == family)]
            if sub.empty:
                continue
            best = sub.sort_values(["val_macro_f1", "val_accuracy"],
                                   ascending=[False, False]).iloc[0]
            rows.append(best.to_dict())

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return
    plot_df.to_csv(paths["fig1_plot_data_csv"], index=False)

    # Narrow canvas so LaTeX downscales less and the text stays readable.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    x = np.arange(len(F.VARIANT_DISPLAY_ORDER))
    width = 0.34

    for ax, metric, panel_title in [(axes[0], "val_accuracy", "Validation accuracy"),
                                    (axes[1], "val_macro_f1", "Validation macro-F1")]:
        hgb_vals, xgb_vals = [], []
        for variant in F.VARIANT_DISPLAY_ORDER:
            hgb = plot_df[(plot_df["feature_variant"] == variant)
                          & (plot_df["classifier_family"] == "HistGradientBoosting")]
            xgb = plot_df[(plot_df["feature_variant"] == variant)
                          & (plot_df["classifier_family"] == "XGBoost")]
            hgb_vals.append(float(hgb.iloc[0][metric]) if not hgb.empty else np.nan)
            xgb_vals.append(float(xgb.iloc[0][metric]) if not xgb.empty else np.nan)

        ax.bar(x - width / 2, hgb_vals, width, label="HGB", color=PLOT_BLUE)
        ax.bar(x + width / 2, xgb_vals, width, label="XGB", color=PLOT_WINE)

        ax.set_title(panel_title, fontsize=9.5, pad=6)
        ax.set_ylabel(panel_title, fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABELS[v] for v in F.VARIANT_DISPLAY_ORDER],
                           fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, frameon=True, loc="upper left")

        vals = [v for v in hgb_vals + xgb_vals if not np.isnan(v)]
        if vals:
            low = max(0.0, min(vals) - 0.01)
            high = min(1.0, max(vals) + 0.01)
            ax.set_ylim(low, high)
            text_offset = (high - low) * 0.025
        else:
            text_offset = 0.001

        for xpos, val in zip(x - width / 2, hgb_vals):
            if not np.isnan(val):
                ax.text(xpos, val + text_offset, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=5.5)
        for xpos, val in zip(x + width / 2, xgb_vals):
            if not np.isnan(val):
                ax.text(xpos, val + text_offset, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=5.5)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot(fig,
              os.path.join(paths["plots_dir"], "fig1_feature_variant_comparison.png"),
              os.path.join(paths["plots_dir"], "fig1_feature_variant_comparison.pdf"))


# ============================================================
# DATA EFFICIENCY
# ============================================================

def run_data_efficiency(selected_classifier_config, selected_classifier_template,
                        variant_data, train_idx, val_idx, sizes_text, seeds_text,
                        csv_path):
    from src.selection.train import metric_report

    sizes = parse_size_list(sizes_text, len(train_idx))
    seeds = parse_seed_list(seeds_text)

    print(SEP)
    print("Running data-efficiency experiment")
    print(SEP)
    print(f"Classifier config: {selected_classifier_config}")
    print(f"Training sizes   : {sizes}")
    print(f"Seeds            : {seeds}\n")

    rows = []
    for variant in F.VARIANT_DISPLAY_ORDER:
        data = variant_data[variant]
        X, y = data["X"], data["y"]
        y_ids = y.map(F.LABEL_TO_ID).values
        X_val = X.iloc[val_idx]
        y_true = [F.ID_TO_LABEL[int(i)] for i in y_ids[val_idx]]

        for size in sizes:
            if size == "full":
                size_label, numeric_size, run_seeds = "full", len(train_idx), [seeds[0]]
            else:
                size_label, numeric_size, run_seeds = str(size), int(size), seeds

            for seed in run_seeds:
                if size == "full":
                    sub_idx = np.array(train_idx, dtype=int)
                else:
                    sub_idx = stratified_subset_indices(train_idx, y, numeric_size, seed)

                start = time.time()
                model = fit_model_on_indices(selected_classifier_template, X, y, sub_idx)
                fit_seconds = time.time() - start

                y_pred = [F.ID_TO_LABEL[int(i)] for i in model.predict(X_val)]
                res = metric_report(y_true, y_pred)

                rows.append({
                    "feature_variant": variant,
                    "feature_variant_display": F.VARIANT_DISPLAY_NAMES.get(variant, variant),
                    "classifier_config": selected_classifier_config,
                    "train_size_label": size_label,
                    "train_size": int(len(sub_idx)),
                    "seed": int(seed),
                    "validation_rows": int(len(val_idx)),
                    "feature_count": int(X.shape[1]),
                    "fit_seconds": float(fit_seconds),
                    "val_accuracy": res["accuracy"],
                    "val_macro_f1": res["macro_f1"],
                    "val_balanced_accuracy": res["balanced_accuracy"],
                })

                print(f"{variant:<20} size={size_label:<6} seed={seed:<4} "
                      f"acc={res['accuracy']:.4f} macro_f1={res['macro_f1']:.4f} "
                      f"time={fit_seconds:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def plot_data_efficiency(df, paths):
    if df.empty:
        return

    # Numeric sizes first, then full.
    ordered_labels = sorted(
        [v for v in df["train_size_label"].unique() if v != "full"], key=lambda x: int(x))
    if "full" in set(df["train_size_label"]):
        ordered_labels.append("full")

    x_positions = np.arange(len(ordered_labels))
    label_to_x = {lab: i for i, lab in enumerate(ordered_labels)}

    summary = (df.groupby(["feature_variant", "train_size_label"], as_index=False)
               .agg(val_macro_f1_mean=("val_macro_f1", "mean"),
                    val_macro_f1_std=("val_macro_f1", "std"),
                    val_accuracy_mean=("val_accuracy", "mean"),
                    val_accuracy_std=("val_accuracy", "std"))).fillna(0.0)
    summary.to_csv(paths["fig2_plot_data_csv"], index=False)

    # Line plots need horizontal space, so this one stays wide.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for ax, mean_col, std_col, panel_title in [
        (axes[0], "val_accuracy_mean", "val_accuracy_std", "Validation accuracy"),
        (axes[1], "val_macro_f1_mean", "val_macro_f1_std", "Validation macro-F1"),
    ]:
        for variant in F.VARIANT_DISPLAY_ORDER:
            sub = summary[summary["feature_variant"] == variant].copy()
            if sub.empty:
                continue
            sub["x"] = sub["train_size_label"].astype(str).map(label_to_x)
            sub = sub.sort_values("x")
            ax.errorbar(sub["x"].values, sub[mean_col].values,
                        yerr=sub[std_col].values, marker="o", capsize=3,
                        linewidth=2, markersize=5,
                        label=VARIANT_LABELS[variant].replace("\n", " "))

        ax.set_title(panel_title, fontsize=14, pad=8)
        ax.set_xlabel("Training examples", fontsize=13)
        ax.set_ylabel(panel_title, fontsize=13)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["Full" if lab == "full" else f"{int(lab)//1000}k"
                            for lab in ordered_labels], fontsize=11)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, frameon=True, loc="best")

        vals = summary[mean_col].dropna().values
        if len(vals) > 0:
            ax.set_ylim(max(0.0, vals.min() - 0.003), min(1.0, vals.max() + 0.003))

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_plot(fig,
              os.path.join(paths["plots_dir"], "fig2_data_efficiency.png"),
              os.path.join(paths["plots_dir"], "fig2_data_efficiency.pdf"))


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def group_permutation_importance(model, X_val, y_val_ids, feature_groups,
                                 n_repeats, random_state):
    baseline_score = macro_f1_from_ids(y_val_ids, model.predict(X_val))
    rng = np.random.default_rng(random_state)
    rows = []

    for group_name, columns in feature_groups.items():
        columns = [c for c in columns if c in X_val.columns]
        if not columns:
            continue

        drops, perm_scores = [], []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            perm = rng.permutation(len(X_perm))
            X_perm.loc[:, columns] = X_perm.iloc[perm][columns].to_numpy()
            score = macro_f1_from_ids(y_val_ids, model.predict(X_perm))
            perm_scores.append(score)
            drops.append(baseline_score - score)

        rows.append({
            "feature_subgroup": group_name,
            "num_features": len(columns),
            "baseline_macro_f1": baseline_score,
            "permuted_macro_f1_mean": float(np.mean(perm_scores)),
            "macro_f1_drop_mean": float(np.mean(drops)),
            "macro_f1_drop_std": float(np.std(drops)),
            "n_repeats": int(n_repeats),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("macro_f1_drop_mean", ascending=False).reset_index(drop=True)
    return df


def run_feature_importance(selected_model, selected_variant, X_val, y_val_ids,
                           n_repeats, random_state, individual_csv, family_csv):
    print(SEP)
    print("Running feature importance")
    print(SEP)
    print(f"Selected variant: {selected_variant}")
    print(f"Validation rows : {len(X_val)}")
    print(f"Feature count   : {X_val.shape[1]}\n")

    scorer = make_scorer(f1_score, average="macro", labels=[0, 1, 2], zero_division=0)
    indiv = permutation_importance(
        selected_model, X_val, y_val_ids, scoring=scorer,
        n_repeats=n_repeats, random_state=random_state, n_jobs=-1)

    indiv_df = pd.DataFrame({
        "feature": list(X_val.columns),
        "feature_display": [display_feature_name(c) for c in X_val.columns],
        "feature_subgroup": [assign_feature_type_family(c) for c in X_val.columns],
        "macro_f1_drop_mean": indiv.importances_mean,
        "macro_f1_drop_std": indiv.importances_std,
    }).sort_values("macro_f1_drop_mean", ascending=False).reset_index(drop=True)
    indiv_df.to_csv(individual_csv, index=False)

    groups = defaultdict(list)
    for col in X_val.columns:
        fam = assign_feature_type_family(col)
        if fam is not None:
            groups[fam].append(col)

    family_df = group_permutation_importance(
        selected_model, X_val, y_val_ids, dict(groups), n_repeats, random_state)
    family_df.to_csv(family_csv, index=False)

    return indiv_df, family_df


def plot_feature_importance_family(family_df, paths):
    if family_df.empty:
        return
    plot_df = family_df.sort_values("macro_f1_drop_mean",
                                    ascending=True).reset_index(drop=True)
    plot_df.to_csv(paths["fig3_subgroup_plot_data_csv"], index=False)

    fig, ax = plt.subplots(figsize=(8.6, max(4.8, 0.55 * len(plot_df))))
    ax.barh(plot_df["feature_subgroup"], plot_df["macro_f1_drop_mean"],
            xerr=plot_df["macro_f1_drop_std"],
            color=viridis_colors(len(plot_df)), ecolor="black", capsize=3)

    ax.set_xlabel("Validation macro-F1 drop after permutation")
    ax.set_ylabel("Feature subgroup")
    ax.set_title("Feature-subgroup importance of selected AVE-LS classifier")
    ax.grid(axis="x", alpha=0.3)

    save_plot(fig,
              os.path.join(paths["plots_dir"], "fig3_feature_subgroup_importance.png"),
              os.path.join(paths["plots_dir"], "fig3_feature_subgroup_importance.pdf"))


def plot_feature_importance_top20(indiv_df, paths):
    if indiv_df.empty:
        return
    top = indiv_df.head(20).sort_values("macro_f1_drop_mean",
                                        ascending=True).reset_index(drop=True)
    top.to_csv(paths["fig3_top20_plot_data_csv"], index=False)

    fig, ax = plt.subplots(figsize=(10.2, 7.4))
    ax.barh(top["feature_display"], top["macro_f1_drop_mean"],
            xerr=top["macro_f1_drop_std"],
            color=viridis_colors(len(top)), ecolor="black", capsize=3)

    ax.set_xlabel("Validation macro-F1 drop after permutation", fontsize=13)
    ax.set_ylabel("Feature", fontsize=14)
    ax.set_title("Top-20 individual feature importances", fontsize=15, pad=10)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=12)

    save_plot(fig,
              os.path.join(paths["plots_dir"], "fig3_feature_importance_top20.png"),
              os.path.join(paths["plots_dir"], "fig3_feature_importance_top20.pdf"))


# ============================================================
# SELECTOR OBJECTIVE ABLATION
# ============================================================

def run_objective_ablation(selected_classifier_config, selected_classifier_template,
                           variant_data, train_idx, val_idx, csv_path, jsonl_path):
    """Label-level target (3 VE labels) against candidate-level target
    (12 prediction-pool outputs), same features, split and classifier."""
    from src.selection.train import metric_report

    variant_name = "full_12_methods"
    data = variant_data[variant_name]
    X, y = data["X"], data["y"]
    meta_rows, outputs_rows = data["meta"], data["outputs"]
    method_keys = F.FEATURE_VARIANTS[variant_name]

    y_label_ids = y.map(F.LABEL_TO_ID).values
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_label, y_val_label = y_label_ids[train_idx], y_label_ids[val_idx]
    y_true_labels = [F.ID_TO_LABEL[int(i)] for i in y_val_label]

    print(SEP)
    print("Running selector objective ablation")
    print(SEP)
    print(f"Feature variant : {variant_name}")
    print(f"Classifier config: {selected_classifier_config}")
    print(f"Label target classes     : {len(F.LABELS)}")
    print(f"Candidate target classes : {len(method_keys)}\n")

    rows, pred_rows = [], []

    # Objective 1: label-level, target = 3 VE labels.
    label_model = prepare_classifier_for_num_classes(
        selected_classifier_template, len(F.LABELS))
    start = time.time()
    label_model.fit(X_train, y_train_label)
    label_fit_seconds = time.time() - start

    label_pred_labels = [F.ID_TO_LABEL[int(i)] for i in label_model.predict(X_val)]
    label_metrics = metric_report(y_true_labels, label_pred_labels)

    rows.append({
        "objective": "label_level",
        "objective_display": "Label-level selector (3 labels)",
        "feature_variant": variant_name,
        "classifier_config": selected_classifier_config,
        "target_classes": len(F.LABELS),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "feature_count": int(X.shape[1]),
        "fit_seconds": float(label_fit_seconds),
        "val_accuracy": label_metrics["accuracy"],
        "val_macro_f1": label_metrics["macro_f1"],
        "val_balanced_accuracy": label_metrics["balanced_accuracy"],
        "selector_target_accuracy": np.nan,
        "target_rule": "gold VE label",
    })

    for local_idx, original_idx in enumerate(val_idx):
        gold = F.normalize_label(y_true_labels[local_idx])
        pred = F.normalize_label(label_pred_labels[local_idx])
        pred_rows.append({
            "objective": "label_level",
            "feature_variant": variant_name,
            "classifier_config": selected_classifier_config,
            "Flickr30K_ID": meta_rows[int(original_idx)]["Flickr30K_ID"],
            "hypothesis": meta_rows[int(original_idx)]["hypothesis"],
            "gold": gold,
            "final_label": pred,
            "selected_candidate": None,
            "selected_candidate_label": None,
            "oracle_candidate": None,
            "oracle_target_reason": None,
            "selector_target_correct": None,
            "final_correct": int(pred == gold),
        })

    print(f"label_level       acc={label_metrics['accuracy']:.4f} "
          f"macro_f1={label_metrics['macro_f1']:.4f} time={label_fit_seconds:.1f}s")

    # Objective 2: candidate-level, target = 12 prediction-pool outputs.
    method_to_target_id = {m: i for i, m in enumerate(method_keys)}
    target_id_to_method = {i: m for m, i in method_to_target_id.items()}

    oracle_methods, oracle_reasons = [], []
    for i, outputs in enumerate(outputs_rows):
        winner, reason = choose_oracle_candidate_winner(outputs, method_keys, y.iloc[i])
        oracle_methods.append(winner)
        oracle_reasons.append(reason)

    y_candidate_ids = np.array([method_to_target_id[m] for m in oracle_methods], dtype=int)

    candidate_model = prepare_classifier_for_num_classes(
        selected_classifier_template, len(method_keys))
    start = time.time()
    candidate_model.fit(X_train, y_candidate_ids[train_idx])
    candidate_fit_seconds = time.time() - start

    candidate_pred_ids = np.array(
        [int(i) for i in candidate_model.predict(X_val)], dtype=int)
    candidate_target_accuracy = float(
        accuracy_score(y_candidate_ids[val_idx], candidate_pred_ids))

    candidate_pred_labels = []
    for local_idx, original_idx in enumerate(val_idx):
        selected_method = target_id_to_method[int(candidate_pred_ids[local_idx])]
        selected_output = outputs_rows[int(original_idx)][selected_method]
        candidate_pred_labels.append(
            F.normalize_label(selected_output.get("prediction", "neutral")))

    candidate_metrics = metric_report(y_true_labels, candidate_pred_labels)

    rows.append({
        "objective": "candidate_level",
        "objective_display": "Candidate-level selector (12 outputs)",
        "feature_variant": variant_name,
        "classifier_config": selected_classifier_config,
        "target_classes": len(method_keys),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "feature_count": int(X.shape[1]),
        "fit_seconds": float(candidate_fit_seconds),
        "val_accuracy": candidate_metrics["accuracy"],
        "val_macro_f1": candidate_metrics["macro_f1"],
        "val_balanced_accuracy": candidate_metrics["balanced_accuracy"],
        "selector_target_accuracy": candidate_target_accuracy,
        "target_rule": "oracle candidate: correct highest margin; fallback highest gold-label score",
    })

    for local_idx, original_idx in enumerate(val_idx):
        original_idx = int(original_idx)
        gold = F.normalize_label(y_true_labels[local_idx])
        selected_method = target_id_to_method[int(candidate_pred_ids[local_idx])]
        selected_output = outputs_rows[original_idx][selected_method]
        final_label = F.normalize_label(selected_output.get("prediction", "neutral"))
        pred_rows.append({
            "objective": "candidate_level",
            "feature_variant": variant_name,
            "classifier_config": selected_classifier_config,
            "Flickr30K_ID": meta_rows[original_idx]["Flickr30K_ID"],
            "hypothesis": meta_rows[original_idx]["hypothesis"],
            "gold": gold,
            "final_label": final_label,
            "selected_candidate": selected_method,
            "selected_candidate_label": final_label,
            "oracle_candidate": oracle_methods[original_idx],
            "oracle_target_reason": oracle_reasons[original_idx],
            "selector_target_correct": int(selected_method == oracle_methods[original_idx]),
            "final_correct": int(final_label == gold),
        })

    print(f"candidate_level   acc={candidate_metrics['accuracy']:.4f} "
          f"macro_f1={candidate_metrics['macro_f1']:.4f} "
          f"selector_target_acc={candidate_target_accuracy:.4f} "
          f"time={candidate_fit_seconds:.1f}s\n")

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(csv_path, index=False)
    F.write_jsonl(jsonl_path, pred_rows)
    return ablation_df


def plot_objective_ablation(ablation_df, paths):
    if ablation_df.empty:
        return

    order = ["label_level", "candidate_level"]
    plot_df = ablation_df.set_index("objective").loc[order].reset_index()
    plot_df.to_csv(paths["fig4_plot_data_csv"], index=False)

    labels = [OBJECTIVE_LABELS.get(obj, obj) for obj in plot_df["objective"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))

    for ax, metric, panel_title in [(axes[0], "val_accuracy", "Validation accuracy"),
                                    (axes[1], "val_macro_f1", "Validation macro-F1")]:
        vals = plot_df[metric].astype(float).values
        ax.bar(x, vals, width=0.55, color=BAR_COLORS[:len(vals)])

        ax.set_title(panel_title, fontsize=9.5, pad=6)
        ax.set_ylabel(panel_title, fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.3)

        low = max(0.0, min(vals) - 0.01)
        high = min(1.0, max(vals) + 0.01)
        ax.set_ylim(low, high)
        text_offset = (high - low) * 0.025

        for xpos, val in zip(x, vals):
            ax.text(xpos, val + text_offset, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6.5)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot(fig,
              os.path.join(paths["plots_dir"], "fig4_selector_objective_ablation.png"),
              os.path.join(paths["plots_dir"], "fig4_selector_objective_ablation.pdf"))


# ============================================================
# MAIN
# ============================================================

def main():
    from src.selection.classifiers import build_classifier_configs
    from src.selection.train import (
        build_dataset_for_variant,
        build_paths,
        compute_global_common_keys,
        load_all_required_methods,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-name", default="AVE_train_learned_selector_v3")
    ap.add_argument("--validation-size", type=float, default=0.20)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--data-efficiency-sizes",
                    default="5000,10000,15000,20000,25000,30000,35000,full")
    ap.add_argument("--data-efficiency-seeds", default="42,43,44")
    ap.add_argument("--permutation-repeats", type=int, default=5)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["data_efficiency", "importance", "objective"])
    args = ap.parse_args()

    paths = build_paths(args.output_name)
    r = paths["results_dir"]
    paths.update({
        "data_efficiency_csv": os.path.join(r, "data_efficiency_results.csv"),
        "perm_individual_csv": os.path.join(r, "permutation_importance_individual.csv"),
        "perm_family_csv": os.path.join(r, "permutation_importance_feature_subgroup.csv"),
        "objective_ablation_csv": os.path.join(r, "objective_ablation_label_vs_candidate.csv"),
        "objective_ablation_val_jsonl": os.path.join(
            paths["predictions_dir"], "objective_ablation_validation_predictions.jsonl"),
        "fig1_plot_data_csv": os.path.join(r, "fig1_feature_variant_comparison_plot_data.csv"),
        "fig2_plot_data_csv": os.path.join(r, "fig2_data_efficiency_plot_data.csv"),
        "fig3_subgroup_plot_data_csv": os.path.join(
            r, "fig3_feature_subgroup_importance_plot_data.csv"),
        "fig3_top20_plot_data_csv": os.path.join(
            r, "fig3_top20_individual_feature_importance_plot_data.csv"),
        "fig4_plot_data_csv": os.path.join(r, "fig4_selector_objective_ablation_plot_data.csv"),
    })

    with open(paths["selected_config_json"], "r", encoding="utf-8") as f:
        selected_config = json.load(f)

    selected_variant = selected_config["selected_feature_variant"]
    selected_classifier = selected_config["selected_classifier_config"]
    classifier_configs = build_classifier_configs(args.random_state)
    selected_template = classifier_configs[selected_classifier]

    print(SEP)
    print("AVE-LS POST-TRAINING ANALYSES")
    print(SEP)
    print(f"Run       : {paths['output_dir']}")
    print(f"Variant   : {selected_variant}")
    print(f"Classifier: {selected_classifier}\n")

    train_metadata = F.load_dataset_metadata(paths["train_input_jsonl"])
    loaded_methods = load_all_required_methods(paths)
    global_keys = compute_global_common_keys(train_metadata, loaded_methods)

    variant_data = {}
    for variant_name in F.VARIANT_DISPLAY_ORDER:
        X, y, meta, outputs = build_dataset_for_variant(
            variant_name, global_keys, train_metadata, loaded_methods)
        variant_data[variant_name] = {"X": X, "y": y, "meta": meta, "outputs": outputs}

    # Recreate the same fixed stratified split used during training.
    y_global = variant_data[F.VARIANT_DISPLAY_ORDER[0]]["y"]
    train_idx, val_idx = train_test_split(
        np.arange(len(global_keys)),
        test_size=args.validation_size,
        stratify=y_global.map(F.LABEL_TO_ID).values,
        random_state=args.random_state)
    train_idx = np.array(sorted(train_idx), dtype=int)
    val_idx = np.array(sorted(val_idx), dtype=int)

    print(f"Train rows: {len(train_idx)}  Validation rows: {len(val_idx)}\n")

    if "data_efficiency" not in args.skip:
        de_df = run_data_efficiency(
            selected_classifier, selected_template, variant_data,
            train_idx, val_idx, args.data_efficiency_sizes,
            args.data_efficiency_seeds, paths["data_efficiency_csv"])
        plot_data_efficiency(de_df, paths)

    if "importance" not in args.skip:
        data = variant_data[selected_variant]
        y_ids = data["y"].map(F.LABEL_TO_ID).values
        # Refit on the training portion, so importance is measured on data the
        # model has not seen. The saved model was fitted on all rows.
        model = fit_model_on_indices(selected_template, data["X"], data["y"], train_idx)
        indiv_df, family_df = run_feature_importance(
            model, selected_variant, data["X"].iloc[val_idx], y_ids[val_idx],
            args.permutation_repeats, args.random_state,
            paths["perm_individual_csv"], paths["perm_family_csv"])
        plot_feature_importance_family(family_df, paths)
        plot_feature_importance_top20(indiv_df, paths)

    if "objective" not in args.skip:
        ab_df = run_objective_ablation(
            selected_classifier, selected_template, variant_data,
            train_idx, val_idx, paths["objective_ablation_csv"],
            paths["objective_ablation_val_jsonl"])
        plot_objective_ablation(ab_df, paths)

    results_df = pd.read_csv(paths["all_results_csv"])
    plot_feature_variant_comparison(results_df, paths)

    print(f"\nWritten to {paths['output_dir']}")


if __name__ == "__main__":
    main()