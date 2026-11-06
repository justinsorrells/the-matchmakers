"""Fit and evaluate the frozen Milestone 3 actor logistic baseline."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matchmakers-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "matchmakers-cache")
)

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_audit import CORE_RATINGS
from .milestone_common import (
    AUDIT_V5,
    M3_OUTPUT,
    RAW_DATA,
    ContractViolation,
    assert_held_out_wave_separation,
    commit_staged_artifacts,
    frozen_input_hashes,
    load_analysis_frame,
    refuse_existing_artifacts,
    validate_feature_allowlist,
    validate_pinned_environment,
    write_json,
)


C_GRID = (0.01, 0.1, 1.0, 10.0)
SMALL_WAVES = (6, 18, 20)
RESAMPLING_SEED = 4623
PERMUTATION_SEED = 4624
RESAMPLING_REPETITIONS = 10_000
PERMUTATION_REPETITIONS = 2_000

ARTIFACT_NAMES = [
    "actor_outer_predictions.csv",
    "actor_metrics_by_wave.csv",
    "actor_tuning_by_outer_wave.csv",
    "actor_tuning_by_inner_wave.csv",
    "average_precision_by_wave.csv",
    "average_precision_permutations.csv",
    "reliability_table.csv",
    "reliability_plot.png",
    "participant_predictions.csv",
    "participant_metrics_by_wave.csv",
    "resampling_draws.csv",
    "resampling_intervals.csv",
    "m3_summary.json",
    "m3_manifest.json",
]

NAVY = "#0D2D3D"
BLUE = "#3F7C8D"
CORAL = "#C65A36"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DATA)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_V5)
    parser.add_argument("--output-dir", type=Path, default=M3_OUTPUT)
    parser.add_argument(
        "--force", action="store_true", help="Replace existing Milestone 3 artifacts."
    )
    return parser.parse_args()


def build_logistic_pipeline(
    c_value: float,
    features: Sequence[str],
    *,
    audit_dir: Path = AUDIT_V5,
) -> Pipeline:
    """Construct the only estimator allowed for the Milestone 3 actor primary."""
    validate_feature_allowlist(
        features, "core", audit_dir=audit_dir, require_exact=True
    )
    if float(c_value) not in C_GRID:
        raise ContractViolation(f"C must be in the frozen grid {C_GRID}: {c_value}")
    return Pipeline(
        steps=[
            ("standard_scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=float(c_value),
                    l1_ratio=0.0,
                    solver="lbfgs",
                    class_weight=None,
                    max_iter=2_000,
                ),
            ),
        ]
    )


def validate_actor_primary_rows(frame: pd.DataFrame, audit_dir: Path) -> pd.DataFrame:
    features = validate_feature_allowlist(
        CORE_RATINGS, "core", audit_dir=audit_dir, require_exact=True
    )
    core = frame.loc[~frame["core_rating_missing"]].copy()
    if core[features].isna().any().any():
        raise ContractViolation("complete-core actor rows contain missing model features")
    if len(core) != 8_273:
        raise ContractViolation(f"actor primary must contain 8,273 rows; observed={len(core)}")
    if int(core["match"].sum()) != 1_351:
        raise ContractViolation(
            f"actor primary must contain 1,351 positive rows; observed={int(core['match'].sum())}"
        )
    if core["participant_id"].nunique() != 542:
        raise ContractViolation(
            "actor primary must contain 542 complete-core participants; "
            f"observed={core['participant_id'].nunique()}"
        )
    if core["wave"].nunique() != 21:
        raise ContractViolation("actor primary must contain all 21 waves")

    one_sided = pd.read_csv(audit_dir / "one_sided_rows.csv")
    included = set(one_sided["row_index"].astype(int)) & set(core["row_index"].astype(int))
    if len(included) != 10:
        raise ContractViolation(
            f"all ten one-sided wave-5 rows must be in the actor primary; observed={len(included)}"
        )

    invariance = core.groupby("participant_id", sort=False)[features].nunique(dropna=False)
    if (invariance > 1).any().any():
        raise ContractViolation("core actor features vary within a participant")
    return core.sort_values("row_index").reset_index(drop=True)


def choose_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(
        y_true, scores, drop_intermediate=False
    )
    numerator = 2.0 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )
    best_f1 = float(np.max(f1_values))
    tied = thresholds[f1_values == best_f1]
    return float(np.max(tied)), best_f1


def select_logistic_c(mean_auc_by_c: dict[float, float]) -> float:
    """Select the best frozen-grid C, retaining the smaller C on exact ties."""
    if set(mean_auc_by_c) != set(C_GRID):
        raise ContractViolation(
            f"tuning results must cover exactly the frozen C grid: {mean_auc_by_c}"
        )
    best_c = float(C_GRID[0])
    best_score = float(mean_auc_by_c[best_c])
    for c_value in C_GRID[1:]:
        candidate = float(mean_auc_by_c[float(c_value)])
        if candidate > best_score:
            best_c = float(c_value)
            best_score = candidate
    return best_c


def tune_outer_training_data(
    train: pd.DataFrame,
    outer_wave: int,
    *,
    features: Sequence[str],
    audit_dir: Path,
) -> tuple[float, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    x = train[list(features)]
    y = train["match"].to_numpy(dtype=int)
    groups = train["wave"].to_numpy(dtype=int)
    splitter = GroupKFold(n_splits=5, shuffle=False)
    splits = list(splitter.split(x, y, groups=groups))

    summary_records: list[dict[str, Any]] = []
    inner_records: list[dict[str, Any]] = []
    oof_by_c: dict[float, np.ndarray] = {}
    mean_auc_by_c: dict[float, float] = {}

    for c_value in C_GRID:
        oof = np.full(len(train), np.nan, dtype=float)
        inner_fold_by_row = np.full(len(train), -1, dtype=int)
        for inner_fold, (fit_indices, validation_indices) in enumerate(splits):
            fit_waves = groups[fit_indices]
            validation_waves = groups[validation_indices]
            if outer_wave in fit_waves or outer_wave in validation_waves:
                raise ContractViolation(
                    f"outer wave {outer_wave} leaked into inner tuning data"
                )
            if set(fit_waves) & set(validation_waves):
                raise ContractViolation(
                    f"inner wave overlap for outer wave {outer_wave}, fold {inner_fold}"
                )
            pipeline = build_logistic_pipeline(c_value, features, audit_dir=audit_dir)
            pipeline.fit(x.iloc[fit_indices], y[fit_indices])
            oof[validation_indices] = pipeline.predict_proba(x.iloc[validation_indices])[:, 1]
            inner_fold_by_row[validation_indices] = inner_fold

        if np.isnan(oof).any() or (inner_fold_by_row < 0).any():
            raise ContractViolation(f"inner OOF predictions incomplete for outer wave {outer_wave}")

        per_wave_auc: list[float] = []
        for inner_wave in sorted(set(groups)):
            mask = groups == inner_wave
            wave_y = y[mask]
            if np.unique(wave_y).size != 2:
                raise ContractViolation(
                    f"inner validation wave {inner_wave} lacks both outcome classes"
                )
            auc = float(roc_auc_score(wave_y, oof[mask]))
            per_wave_auc.append(auc)
            folds = np.unique(inner_fold_by_row[mask])
            if len(folds) != 1:
                raise ContractViolation(
                    f"wave {inner_wave} spans multiple GroupKFold validation folds"
                )
            inner_records.append(
                {
                    "outer_wave": int(outer_wave),
                    "c": float(c_value),
                    "inner_wave": int(inner_wave),
                    "inner_fold": int(folds[0]),
                    "validation_rows": int(mask.sum()),
                    "positive_rows": int(wave_y.sum()),
                    "roc_auc": auc,
                    "roc_auc_minus_0_5": auc - 0.5,
                }
            )
        mean_auc = float(np.mean(per_wave_auc))
        oof_by_c[float(c_value)] = oof
        mean_auc_by_c[float(c_value)] = mean_auc

    best_c = select_logistic_c(mean_auc_by_c)

    for c_value in C_GRID:
        summary_records.append(
            {
                "outer_wave": int(outer_wave),
                "c": float(c_value),
                "mean_inner_wave_roc_auc": mean_auc_by_c[float(c_value)],
                "selected": float(c_value) == best_c,
                "tie_rule": "smaller C wins exact ties",
            }
        )
    return best_c, oof_by_c[best_c], summary_records, inner_records


def run_outer_validation(
    core: pd.DataFrame, audit_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = list(CORE_RATINGS)
    prediction_parts: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    tuning_records: list[dict[str, Any]] = []
    inner_records: list[dict[str, Any]] = []

    for outer_wave in sorted(core["wave"].unique()):
        train = core[core["wave"] != outer_wave].copy()
        test = core[core["wave"] == outer_wave].copy()
        assert_held_out_wave_separation(train["wave"], test["wave"], int(outer_wave))
        if train["wave"].nunique() != 20:
            raise ContractViolation("each LOWO outer training set must contain 20 waves")

        selected_c, selected_inner_oof, tuning, inner = tune_outer_training_data(
            train,
            int(outer_wave),
            features=features,
            audit_dir=audit_dir,
        )
        tuning_records.extend(tuning)
        inner_records.extend(inner)
        threshold, inner_f1 = choose_f1_threshold(
            train["match"].to_numpy(dtype=int), selected_inner_oof
        )

        pipeline = build_logistic_pipeline(selected_c, features, audit_dir=audit_dir)
        pipeline.fit(train[features], train["match"].to_numpy(dtype=int))
        scores = pipeline.predict_proba(test[features])[:, 1]
        predictions = (scores >= threshold).astype(int)
        y_test = test["match"].to_numpy(dtype=int)
        training_prevalence = float(train["match"].mean())
        baseline_scores = np.full(len(test), training_prevalence, dtype=float)

        wave_predictions = test[
            ["row_index", "wave", "participant_id", "actor_fold", "match"]
        ].copy()
        wave_predictions["selected_c"] = selected_c
        wave_predictions["f1_threshold"] = threshold
        wave_predictions["score"] = scores
        wave_predictions["predicted_label"] = predictions
        wave_predictions["outer_training_prevalence"] = training_prevalence
        prediction_parts.append(wave_predictions)

        auc = float(roc_auc_score(y_test, scores))
        metric_records.append(
            {
                "wave": int(outer_wave),
                "test_rows": int(len(test)),
                "positive_rows": int(y_test.sum()),
                "test_prevalence": float(y_test.mean()),
                "training_rows": int(len(train)),
                "training_waves": int(train["wave"].nunique()),
                "outer_training_prevalence": training_prevalence,
                "selected_c": selected_c,
                "selected_inner_mean_roc_auc": float(
                    next(
                        row["mean_inner_wave_roc_auc"]
                        for row in tuning
                        if row["selected"]
                    )
                ),
                "f1_threshold": threshold,
                "inner_oof_f1": inner_f1,
                "roc_auc": auc,
                "roc_auc_minus_0_5": auc - 0.5,
                "average_precision": float(average_precision_score(y_test, scores)),
                "log_loss": float(log_loss(y_test, scores, labels=[0, 1])),
                "brier_score": float(brier_score_loss(y_test, scores)),
                "training_prevalence_log_loss": float(
                    log_loss(y_test, baseline_scores, labels=[0, 1])
                ),
                "training_prevalence_brier_score": float(
                    brier_score_loss(y_test, baseline_scores)
                ),
                "model_f1": float(f1_score(y_test, predictions, zero_division=0)),
                "model_precision": float(
                    precision_score(y_test, predictions, zero_division=0)
                ),
                "model_recall": float(recall_score(y_test, predictions, zero_division=0)),
                "majority_negative_f1": float(
                    f1_score(y_test, np.zeros(len(test), dtype=int), zero_division=0)
                ),
                "all_positive_f1": float(
                    f1_score(y_test, np.ones(len(test), dtype=int), zero_division=0)
                ),
            }
        )

    outer_predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    if len(outer_predictions) != len(core) or outer_predictions["row_index"].duplicated().any():
        raise ContractViolation("every complete-core row must receive one outer prediction")
    return (
        outer_predictions,
        pd.DataFrame.from_records(metric_records).sort_values("wave").reset_index(drop=True),
        pd.DataFrame.from_records(tuning_records)
        .sort_values(["outer_wave", "c"])
        .reset_index(drop=True),
        pd.DataFrame.from_records(inner_records)
        .sort_values(["outer_wave", "c", "inner_wave"])
        .reset_index(drop=True),
    )


def build_reliability_table(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions[["score", "match"]].copy()
    work["quantile_bin"] = pd.qcut(
        work["score"], q=10, labels=False, duplicates="raise"
    ).astype(int)
    table = (
        work.groupby("quantile_bin", sort=True)
        .agg(
            rows=("match", "size"),
            positive_rows=("match", "sum"),
            mean_predicted_probability=("score", "mean"),
            observed_match_rate=("match", "mean"),
            minimum_score=("score", "min"),
            maximum_score=("score", "max"),
        )
        .reset_index()
    )
    table["quantile_bin"] += 1
    if len(table) != 10:
        raise ContractViolation(f"reliability plot requires 10 quantile bins; observed={len(table)}")
    return table


def plot_reliability(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.6), layout="constrained")
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", label="Perfect reliability")
    ax.plot(
        table["mean_predicted_probability"],
        table["observed_match_rate"],
        color=NAVY,
        marker="o",
        linewidth=2,
        label="Actor logistic outer predictions",
    )
    for row in table.itertuples(index=False):
        ax.annotate(
            str(row.quantile_bin),
            (row.mean_predicted_probability, row.observed_match_rate),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    upper = max(
        0.3,
        float(table[["mean_predicted_probability", "observed_match_rate"]].max().max())
        * 1.12,
    )
    ax.set(
        title="Pooled reliability of held-out actor probabilities",
        xlabel="Mean predicted probability",
        ylabel="Observed mutual-match rate",
        xlim=(0, upper),
        ylim=(0, upper),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def actor_average_precision_permutation(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    observed_records: list[dict[str, float | int]] = []
    wave_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for wave, group in predictions.groupby("wave", sort=True):
        participants = (
            group.groupby("participant_id", sort=True)["score"]
            .agg(["min", "max", "mean"])
            .reset_index()
        )
        if not np.allclose(participants["min"], participants["max"], rtol=0, atol=1e-14):
            raise ContractViolation("actor scores must be constant within participant")
        participant_index = pd.Index(participants["participant_id"])
        codes = participant_index.get_indexer(group["participant_id"])
        if (codes < 0).any():
            raise ContractViolation("participant score coding failed")
        y = group["match"].to_numpy(dtype=int)
        scores = group["score"].to_numpy(dtype=float)
        observed_ap = float(average_precision_score(y, scores))
        observed_records.append(
            {
                "wave": int(wave),
                "directed_rows": int(len(group)),
                "positive_rows": int(y.sum()),
                "participants": int(len(participants)),
                "average_precision": observed_ap,
            }
        )
        wave_inputs.append(
            (y, participants["mean"].to_numpy(dtype=float), codes.astype(int))
        )

    observed = pd.DataFrame.from_records(observed_records)
    observed_mean = float(observed["average_precision"].mean())
    rng = np.random.default_rng(PERMUTATION_SEED)
    null = np.empty(PERMUTATION_REPETITIONS, dtype=float)
    for replicate in range(PERMUTATION_REPETITIONS):
        per_wave = []
        for y, participant_scores, codes in wave_inputs:
            permuted = rng.permutation(participant_scores)
            per_wave.append(float(average_precision_score(y, permuted[codes])))
        null[replicate] = float(np.mean(per_wave))

    permutation_table = pd.DataFrame(
        {
            "replicate": np.arange(1, PERMUTATION_REPETITIONS + 1, dtype=int),
            "null_mean_wave_average_precision": null,
        }
    )
    exceedances = int(np.count_nonzero(null >= observed_mean))
    summary: dict[str, float | int] = {
        "observed_mean_wave_average_precision": observed_mean,
        "null_mean": float(null.mean()),
        "observed_minus_null_mean": observed_mean - float(null.mean()),
        "null_at_least_observed": exceedances,
        "p_value": float((1 + exceedances) / (PERMUTATION_REPETITIONS + 1)),
        "permutations": PERMUTATION_REPETITIONS,
        "seed": PERMUTATION_SEED,
    }
    return observed, permutation_table, summary


def _log_likelihood_improvement(
    successes: np.ndarray,
    failures: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    epsilon = np.finfo(float).eps
    p = np.clip(predicted, epsilon, 1.0 - epsilon)
    b = np.clip(baseline, epsilon, 1.0 - epsilon)
    return successes * np.log(p / b) + failures * np.log((1.0 - p) / (1.0 - b))


def build_participant_results(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    participant = (
        predictions.groupby(["wave", "participant_id"], sort=True)
        .agg(
            trials=("match", "size"),
            successes=("match", "sum"),
            predicted_rate=("score", "mean"),
            minimum_score=("score", "min"),
            maximum_score=("score", "max"),
            outer_training_prevalence=("outer_training_prevalence", "first"),
        )
        .reset_index()
    )
    if not np.allclose(
        participant["minimum_score"], participant["maximum_score"], rtol=0, atol=1e-14
    ):
        raise ContractViolation("participant outer predictions are not constant across dates")
    participant["observed_rate"] = participant["successes"] / participant["trials"]
    participant["brier_improvement"] = (
        (participant["observed_rate"] - participant["outer_training_prevalence"]) ** 2
        - (participant["observed_rate"] - participant["predicted_rate"]) ** 2
    )
    log_likelihood_improvement = _log_likelihood_improvement(
        participant["successes"].to_numpy(dtype=float),
        (participant["trials"] - participant["successes"]).to_numpy(dtype=float),
        participant["predicted_rate"].to_numpy(dtype=float),
        participant["outer_training_prevalence"].to_numpy(dtype=float),
    )
    participant["deviance_improvement"] = 2.0 * log_likelihood_improvement

    wave_records: list[dict[str, Any]] = []
    for wave, group in participant.groupby("wave", sort=True):
        predicted_constant = group["predicted_rate"].nunique(dropna=False) <= 1
        observed_constant = group["observed_rate"].nunique(dropna=False) <= 1
        if predicted_constant or observed_constant:
            correlation = np.nan
            reasons = []
            if predicted_constant:
                reasons.append("predictions constant")
            if observed_constant:
                reasons.append("observed rates constant")
            excluded_reason = "; ".join(reasons)
        else:
            predicted_ranks = group["predicted_rate"].rank(method="average")
            observed_ranks = group["observed_rate"].rank(method="average")
            correlation = float(predicted_ranks.corr(observed_ranks))
            excluded_reason = ""
        wave_records.append(
            {
                "wave": int(wave),
                "participants": int(len(group)),
                "spearman_correlation": correlation,
                "excluded_from_spearman": bool(predicted_constant or observed_constant),
                "exclusion_reason": excluded_reason,
                "mean_brier_improvement": float(group["brier_improvement"].mean()),
                "sum_deviance_improvement": float(group["deviance_improvement"].sum()),
            }
        )
    by_wave = pd.DataFrame.from_records(wave_records)

    row_log_improvement = _log_likelihood_improvement(
        predictions["match"].to_numpy(dtype=float),
        1.0 - predictions["match"].to_numpy(dtype=float),
        predictions["score"].to_numpy(dtype=float),
        predictions["outer_training_prevalence"].to_numpy(dtype=float),
    )
    deviance = float(participant["deviance_improvement"].sum())
    twice_row_log_improvement = float(2.0 * row_log_improvement.sum())
    if not np.isclose(deviance, twice_row_log_improvement, rtol=1e-12, atol=1e-10):
        raise ContractViolation(
            "participant deviance improvement does not equal twice the summed row "
            f"log-loss improvement: {deviance} != {twice_row_log_improvement}"
        )

    eligible = by_wave.loc[~by_wave["excluded_from_spearman"], "spearman_correlation"]
    summary: dict[str, float | int] = {
        "participants": int(len(participant)),
        "spearman_eligible_waves": int(len(eligible)),
        "spearman_excluded_waves": int(by_wave["excluded_from_spearman"].sum()),
        "mean_wave_spearman": float(eligible.mean()),
        "equal_participant_mean_brier_improvement": float(
            participant["brier_improvement"].mean()
        ),
        "sum_participant_deviance_improvement": deviance,
        "twice_summed_row_log_loss_improvement": twice_row_log_improvement,
        "deviance_identity_absolute_difference": abs(
            deviance - twice_row_log_improvement
        ),
    }
    participant = participant.drop(columns=["minimum_score", "maximum_score"])
    return participant, by_wave, summary


def whole_wave_resampling(
    actor_by_wave: pd.DataFrame,
    participant: pd.DataFrame,
    participant_by_wave: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    waves = actor_by_wave["wave"].to_numpy(dtype=int)
    if not np.array_equal(waves, np.sort(waves)) or len(waves) != 21:
        raise ContractViolation("resampling requires the 21 sorted stored wave results")
    rng = np.random.default_rng(RESAMPLING_SEED)
    sampled_indices = rng.integers(
        0, len(waves), size=(RESAMPLING_REPETITIONS, len(waves))
    )

    actor_values = actor_by_wave["roc_auc_minus_0_5"].to_numpy(dtype=float)
    actor_draws = actor_values[sampled_indices].mean(axis=1)

    wave_metrics = participant_by_wave.set_index("wave").reindex(waves)
    spearman_values = wave_metrics["spearman_correlation"].to_numpy(dtype=float)
    spearman_draws = np.nanmean(spearman_values[sampled_indices], axis=1)

    participant_wave = (
        participant.groupby("wave", sort=True)
        .agg(
            participants=("participant_id", "size"),
            brier_sum=("brier_improvement", "sum"),
            deviance_sum=("deviance_improvement", "sum"),
        )
        .reindex(waves)
    )
    participant_counts = participant_wave["participants"].to_numpy(dtype=float)
    brier_sums = participant_wave["brier_sum"].to_numpy(dtype=float)
    deviance_sums = participant_wave["deviance_sum"].to_numpy(dtype=float)
    brier_draws = brier_sums[sampled_indices].sum(axis=1) / participant_counts[
        sampled_indices
    ].sum(axis=1)
    deviance_draws = deviance_sums[sampled_indices].sum(axis=1)

    draws = pd.DataFrame(
        {
            "replicate": np.arange(1, RESAMPLING_REPETITIONS + 1, dtype=int),
            "actor_mean_wave_roc_auc_minus_0_5": actor_draws,
            "participant_mean_wave_spearman": spearman_draws,
            "participant_equal_weight_brier_improvement": brier_draws,
            "participant_sum_deviance_improvement": deviance_draws,
        }
    )
    estimates = {
        "actor_mean_wave_roc_auc_minus_0_5": float(actor_values.mean()),
        "participant_mean_wave_spearman": float(np.nanmean(spearman_values)),
        "participant_equal_weight_brier_improvement": float(
            participant["brier_improvement"].mean()
        ),
        "participant_sum_deviance_improvement": float(
            participant["deviance_improvement"].sum()
        ),
    }
    intervals = []
    for metric, estimate in estimates.items():
        lower, upper = np.percentile(draws[metric], [2.5, 97.5])
        intervals.append(
            {
                "metric": metric,
                "estimate": estimate,
                "lower_95_percentile": float(lower),
                "upper_95_percentile": float(upper),
                "resamples": RESAMPLING_REPETITIONS,
                "seed": RESAMPLING_SEED,
                "cluster": "whole wave",
            }
        )
    return draws, pd.DataFrame.from_records(intervals)


def build_summary(
    predictions: pd.DataFrame,
    actor_by_wave: pd.DataFrame,
    participant_summary: dict[str, float | int],
    intervals: pd.DataFrame,
    average_precision_summary: dict[str, float | int],
) -> dict[str, Any]:
    interval_lookup = intervals.set_index("metric").to_dict(orient="index")
    primary = float(actor_by_wave["roc_auc_minus_0_5"].mean())
    remaining = actor_by_wave[~actor_by_wave["wave"].isin(SMALL_WAVES)]
    return {
        "milestone": 3,
        "contract": {
            "rows": int(len(predictions)),
            "participants": int(predictions["participant_id"].nunique()),
            "waves": int(predictions["wave"].nunique()),
            "positive_rows": int(predictions["match"].sum()),
            "features": list(CORE_RATINGS),
            "model": "StandardScaler + L2 LogisticRegression",
            "c_grid": list(C_GRID),
            "outer_validation": "leave-one-wave-out",
            "inner_validation": "GroupKFold(n_splits=5, shuffle=False) over training waves",
        },
        "actor_primary": {
            "mean_wave_roc_auc_minus_0_5": primary,
            "whole_wave_percentile_interval": interval_lookup[
                "actor_mean_wave_roc_auc_minus_0_5"
            ],
            "small_wave_sensitivity": {
                "excluded_waves": list(SMALL_WAVES),
                "stored_wave_scores_used": int(len(remaining)),
                "refit": False,
                "mean_wave_roc_auc_minus_0_5": float(
                    remaining["roc_auc_minus_0_5"].mean()
                ),
            },
        },
        "actor_probability": {
            "mean_wave_log_loss": float(actor_by_wave["log_loss"].mean()),
            "mean_wave_brier_score": float(actor_by_wave["brier_score"].mean()),
            "mean_wave_training_prevalence_log_loss": float(
                actor_by_wave["training_prevalence_log_loss"].mean()
            ),
            "mean_wave_training_prevalence_brier_score": float(
                actor_by_wave["training_prevalence_brier_score"].mean()
            ),
            "reliability_bins": 10,
        },
        "actor_classification": {
            "mean_wave_model_f1": float(actor_by_wave["model_f1"].mean()),
            "mean_wave_all_positive_f1": float(
                actor_by_wave["all_positive_f1"].mean()
            ),
            "mean_wave_majority_negative_f1": float(
                actor_by_wave["majority_negative_f1"].mean()
            ),
            "threshold_source": "selected-C uncalibrated inner OOF predictions",
            "threshold_tie_rule": "highest threshold wins exact F1 ties",
        },
        "actor_average_precision": average_precision_summary,
        "participant_primary": {
            **participant_summary,
            "whole_wave_percentile_intervals": {
                metric: interval_lookup[metric]
                for metric in (
                    "participant_mean_wave_spearman",
                    "participant_equal_weight_brier_improvement",
                    "participant_sum_deviance_improvement",
                )
            },
        },
    }


def build_milestone_3(
    *,
    input_path: Path = RAW_DATA,
    audit_dir: Path = AUDIT_V5,
    output_dir: Path = M3_OUTPUT,
    force: bool = False,
) -> list[Path]:
    environment = validate_pinned_environment()
    refuse_existing_artifacts(output_dir, ARTIFACT_NAMES, force)
    frame = load_analysis_frame(input_path, audit_dir)
    core = validate_actor_primary_rows(frame, audit_dir)

    predictions, actor_by_wave, tuning, inner = run_outer_validation(core, audit_dir)
    reliability = build_reliability_table(predictions)
    ap_by_wave, ap_permutations, ap_summary = actor_average_precision_permutation(predictions)
    participant, participant_by_wave, participant_summary = build_participant_results(
        predictions
    )
    resampling_draws, intervals = whole_wave_resampling(
        actor_by_wave, participant, participant_by_wave
    )
    summary = build_summary(
        predictions, actor_by_wave, participant_summary, intervals, ap_summary
    )

    with TemporaryDirectory(prefix="matchmakers-m3-") as temporary:
        stage = Path(temporary)
        predictions.to_csv(stage / "actor_outer_predictions.csv", index=False)
        actor_by_wave.to_csv(stage / "actor_metrics_by_wave.csv", index=False)
        tuning.to_csv(stage / "actor_tuning_by_outer_wave.csv", index=False)
        inner.to_csv(stage / "actor_tuning_by_inner_wave.csv", index=False)
        ap_by_wave.to_csv(stage / "average_precision_by_wave.csv", index=False)
        ap_permutations.to_csv(stage / "average_precision_permutations.csv", index=False)
        reliability.to_csv(stage / "reliability_table.csv", index=False)
        participant.to_csv(stage / "participant_predictions.csv", index=False)
        participant_by_wave.to_csv(stage / "participant_metrics_by_wave.csv", index=False)
        resampling_draws.to_csv(stage / "resampling_draws.csv", index=False)
        intervals.to_csv(stage / "resampling_intervals.csv", index=False)
        plot_reliability(reliability, stage / "reliability_plot.png")
        write_json(stage / "m3_summary.json", summary)
        manifest = {
            "milestone": 3,
            "environment": environment,
            "frozen_input_sha256": frozen_input_hashes(input_path, audit_dir),
            "randomness": {
                "folds_and_stochastic_estimators": 4622,
                "resampling": RESAMPLING_SEED,
                "permutations": PERMUTATION_SEED,
                "note": "LOWO and GroupKFold are deterministic; logistic lbfgs is non-stochastic.",
            },
            "feature_contract": {
                "allowlist": "core",
                "features": list(CORE_RATINGS),
                "policy_file": "output/audit/v5/prohibited_features.json",
            },
            "fold_contract": {
                "participant_ids_and_actor_folds": "output/audit/v5/participant_assignments.csv",
                "outer": "leave-one-wave-out over the 21 observed waves",
                "inner": "GroupKFold(n_splits=5, shuffle=False) over each outer training set's 20 waves",
                "actor_folds_recomputed": False,
            },
            "source_columns": {
                "speeddating.arff": ["row_index", "wave", *CORE_RATINGS, "match"],
                "output/audit/v5/participant_assignments.csv": [
                    "row_index",
                    "wave",
                    "participant_id",
                    "actor_fold",
                    "match",
                    "core_rating_missing",
                ],
                "output/audit/v5/one_sided_rows.csv": ["row_index", "wave"],
                "output/audit/v5/prohibited_features.json": [
                    "feature_allowlists.core",
                    "categorical_d_columns",
                    "prohibited_predictive_features",
                ],
            },
            "artifacts": ARTIFACT_NAMES,
        }
        write_json(stage / "m3_manifest.json", manifest)
        return commit_staged_artifacts(stage, output_dir, ARTIFACT_NAMES)


def main() -> None:
    args = parse_args()
    paths = build_milestone_3(
        input_path=args.input,
        audit_dir=args.audit_dir,
        output_dir=args.output_dir,
        force=args.force,
    )
    print("wrote Milestone 3 artifacts:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
