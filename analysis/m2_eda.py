"""Build the six frozen Milestone 2 exploratory figures and their source tables."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

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

from .data_audit import CORE_RATINGS
from .milestone_common import (
    AUDIT_V5,
    M2_OUTPUT,
    RAW_DATA,
    commit_staged_artifacts,
    frozen_input_hashes,
    load_analysis_frame,
    refuse_existing_artifacts,
    validate_pinned_environment,
    write_json,
)


ARTIFACT_NAMES = [
    "01_wave_prevalence.csv",
    "01_wave_prevalence.png",
    "01_wave_prevalence_without_wave12.csv",
    "01_wave_prevalence_without_wave12.png",
    "02_participant_match_counts.csv",
    "02_participant_match_counts.png",
    "02_participant_match_counts_without_wave12.csv",
    "02_participant_match_counts_without_wave12.png",
    "03_core_rating_distributions.csv",
    "03_core_rating_distributions.png",
    "04_core_missing_outcome_rates.csv",
    "04_core_missing_outcome_rates.png",
    "05_early_wave_expectation_coverage.csv",
    "05_early_wave_expectation_coverage.png",
    "06_reciprocal_pair_status.csv",
    "06_reciprocal_pair_status.png",
    "m2_manifest.json",
]

NAVY = "#0D2D3D"
BLUE = "#3F7C8D"
CORAL = "#C65A36"
CREAM = "#F5EFE5"
GRAY = "#5B6770"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DATA)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_V5)
    parser.add_argument("--output-dir", type=Path, default=M2_OUTPUT)
    parser.add_argument(
        "--force", action="store_true", help="Replace existing Milestone 2 artifacts."
    )
    return parser.parse_args()


def apply_style(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wave_prevalence_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = (
        frame.groupby("wave", sort=True)["match"]
        .agg(directed_rows="size", positive_rows="sum", prevalence="mean")
        .reset_index()
    )
    table["wave"] = table["wave"].astype(int)
    table["positive_rows"] = table["positive_rows"].astype(int)
    return table


def plot_wave_prevalence(table: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    ax.bar(table["wave"], table["prevalence"], color=BLUE, width=0.75)
    ax.axhline(
        table["positive_rows"].sum() / table["directed_rows"].sum(),
        color=CORAL,
        linewidth=1.8,
        label="Row-weighted overall prevalence",
    )
    ax.set(title=title, xlabel="Wave", ylabel="Mutual-match prevalence")
    ax.set_xticks(table["wave"])
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylim(0, max(0.35, float(table["prevalence"].max()) * 1.12))
    ax.legend(frameon=False, loc="upper left")
    apply_style(ax)
    save_figure(fig, path)


def participant_match_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = (
        frame.groupby(["wave", "participant_id"], sort=True)["match"]
        .agg(dates="size", matches="sum", match_rate="mean")
        .reset_index()
    )
    table[["wave", "dates", "matches"]] = table[["wave", "dates", "matches"]].astype(int)
    return table


def plot_participant_matches(table: pd.DataFrame, path: Path, title: str) -> None:
    maximum = int(table["matches"].max())
    counts = table["matches"].value_counts().reindex(range(maximum + 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.bar(counts.index, counts.values, color=NAVY, width=0.8)
    ax.set(title=title, xlabel="Participant mutual-match count", ylabel="Participants")
    ax.set_xticks(counts.index)
    apply_style(ax)
    save_figure(fig, path)


def core_distribution_table(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for feature in CORE_RATINGS:
        observed = frame[feature].dropna()
        counts = observed.value_counts().sort_index()
        for rating, count in counts.items():
            records.append(
                {
                    "feature": feature,
                    "rating": float(rating),
                    "count": int(count),
                    "proportion_of_observed": float(count / len(observed)),
                    "observed_rows": int(len(observed)),
                    "missing_rows": int(frame[feature].isna().sum()),
                }
            )
    return pd.DataFrame.from_records(records)


def plot_core_distributions(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.8), sharey=True, layout="constrained")
    for ax, feature in zip(axes, CORE_RATINGS, strict=True):
        part = table[table["feature"] == feature]
        ax.bar(part["rating"], part["proportion_of_observed"], color=BLUE, width=0.75)
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("Rating")
        ax.set_xticks([1, 3, 5, 7, 9, 10])
        apply_style(ax)
    axes[0].set_ylabel("Proportion of observed rows")
    fig.suptitle("Core self-rating distributions")
    save_figure(fig, path)


def core_missing_outcome_table(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame["core_rating_missing"].map(
        {False: "Complete core ratings", True: "Missing core profile"}
    )
    table = (
        frame.assign(core_status=labels)
        .groupby("core_status", sort=False)["match"]
        .agg(directed_rows="size", positive_rows="sum", match_rate="mean")
        .reset_index()
    )
    table["positive_rows"] = table["positive_rows"].astype(int)
    return table


def plot_core_missing_outcomes(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8), layout="constrained")
    bars = ax.bar(table["core_status"], table["match_rate"], color=[NAVY, CORAL])
    for bar, row in zip(bars, table.itertuples(index=False), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{row.match_rate:.1%}\n(n={row.directed_rows:,})",
            ha="center",
            va="bottom",
        )
    ax.set(title="Outcome rate by core-rating completeness", ylabel="Mutual-match prevalence")
    ax.set_ylim(0, max(0.35, float(table["match_rate"].max()) * 1.28))
    apply_style(ax)
    save_figure(fig, path)


def expectation_coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    early = frame[frame["wave"].between(1, 5)].copy()
    early["expectation_observed"] = early["expected_num_interested_in_me"].notna()
    records: list[dict[str, float | int]] = []
    for wave, group in early.groupby("wave", sort=True):
        observed = group["expectation_observed"]
        missing_matches = group.loc[~observed, "match"]
        observed_matches = group.loc[observed, "match"]
        records.append(
            {
                "wave": int(wave),
                "directed_rows": int(len(group)),
                "expectation_observed_rows": int(observed.sum()),
                "expectation_missing_rows": int((~observed).sum()),
                "coverage": float(observed.mean()),
                "observed_expectation_match_rate": float(observed_matches.mean()),
                "missing_expectation_match_rate": (
                    float(missing_matches.mean()) if len(missing_matches) else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_expectation_coverage(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    bars = ax.bar(table["wave"], table["coverage"], color=BLUE, width=0.7)
    for bar, row in zip(bars, table.itertuples(index=False), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(0.02, bar.get_height() - 0.08),
            f"{row.expectation_observed_rows}/{row.directed_rows}",
            ha="center",
            va="top",
            color="white" if bar.get_height() > 0.18 else NAVY,
            fontsize=9,
        )
    ax.set(
        title="Early-wave expectation-response coverage",
        xlabel="Wave",
        ylabel="Observed expected-interest response",
    )
    ax.set_xticks(table["wave"])
    ax.set_ylim(0, 1.05)
    apply_style(ax)
    save_figure(fig, path)


def reciprocal_status_table(audit_dir: Path) -> pd.DataFrame:
    pairs = pd.read_csv(audit_dir / "reciprocal_pairs.csv")
    one_sided = pd.read_csv(audit_dir / "one_sided_rows.csv")
    strict = int((pairs["pairing_status"] == "strict").sum())
    recovered = int((pairs["pairing_status"] == "recovered_postdate_rating_outlier").sum())
    return pd.DataFrame(
        [
            {
                "status": "Strict reciprocal",
                "reciprocal_pairs": strict,
                "directed_rows": strict * 2,
            },
            {
                "status": "Recovered rating outlier",
                "reciprocal_pairs": recovered,
                "directed_rows": recovered * 2,
            },
            {
                "status": "One-sided record",
                "reciprocal_pairs": 0,
                "directed_rows": int(len(one_sided)),
            },
        ]
    )


def plot_reciprocal_status(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    bars = ax.barh(table["status"], table["directed_rows"], color=[NAVY, BLUE, CORAL])
    ax.set_xscale("log")
    ax.set(title="Directed records by reciprocal-pair status", xlabel="Directed rows (log scale)")
    for bar, value in zip(bars, table["directed_rows"], strict=True):
        ax.text(value * 1.08, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center")
    apply_style(ax, grid_axis="x")
    save_figure(fig, path)


def build_milestone_2(
    *,
    input_path: Path = RAW_DATA,
    audit_dir: Path = AUDIT_V5,
    output_dir: Path = M2_OUTPUT,
    force: bool = False,
) -> list[Path]:
    environment = validate_pinned_environment()
    refuse_existing_artifacts(output_dir, ARTIFACT_NAMES, force)
    frame = load_analysis_frame(input_path, audit_dir)

    wave = wave_prevalence_table(frame)
    wave_without_12 = wave[wave["wave"] != 12].reset_index(drop=True)
    participants = participant_match_table(frame)
    participants_without_12 = participants[participants["wave"] != 12].reset_index(drop=True)
    core = core_distribution_table(frame)
    core_missing = core_missing_outcome_table(frame)
    expectation = expectation_coverage_table(frame)
    reciprocal = reciprocal_status_table(audit_dir)

    with TemporaryDirectory(prefix="matchmakers-m2-") as temporary:
        stage = Path(temporary)
        wave.to_csv(stage / "01_wave_prevalence.csv", index=False)
        wave_without_12.to_csv(stage / "01_wave_prevalence_without_wave12.csv", index=False)
        participants.to_csv(stage / "02_participant_match_counts.csv", index=False)
        participants_without_12.to_csv(
            stage / "02_participant_match_counts_without_wave12.csv", index=False
        )
        core.to_csv(stage / "03_core_rating_distributions.csv", index=False)
        core_missing.to_csv(stage / "04_core_missing_outcome_rates.csv", index=False)
        expectation.to_csv(stage / "05_early_wave_expectation_coverage.csv", index=False)
        reciprocal.to_csv(stage / "06_reciprocal_pair_status.csv", index=False)

        plot_wave_prevalence(
            wave, stage / "01_wave_prevalence.png", "Mutual-match prevalence by wave"
        )
        plot_wave_prevalence(
            wave_without_12,
            stage / "01_wave_prevalence_without_wave12.png",
            "Mutual-match prevalence by wave (wave 12 excluded)",
        )
        plot_participant_matches(
            participants,
            stage / "02_participant_match_counts.png",
            "Participant mutual-match counts",
        )
        plot_participant_matches(
            participants_without_12,
            stage / "02_participant_match_counts_without_wave12.png",
            "Participant mutual-match counts (wave 12 excluded)",
        )
        plot_core_distributions(core, stage / "03_core_rating_distributions.png")
        plot_core_missing_outcomes(core_missing, stage / "04_core_missing_outcome_rates.png")
        plot_expectation_coverage(expectation, stage / "05_early_wave_expectation_coverage.png")
        plot_reciprocal_status(reciprocal, stage / "06_reciprocal_pair_status.png")

        manifest = {
            "milestone": 2,
            "figure_concepts": 6,
            "rendered_figure_files": 8,
            "without_wave_12_variants": [
                "01_wave_prevalence",
                "02_participant_match_counts",
            ],
            "environment": environment,
            "frozen_input_sha256": frozen_input_hashes(input_path, audit_dir),
            "figures": [
                {
                    "id": "01_wave_prevalence",
                    "source_table": "01_wave_prevalence.csv",
                    "source_columns": {"speeddating.arff": ["wave", "match"]},
                    "variant": "01_wave_prevalence_without_wave12.png",
                    "variant_source_table": "01_wave_prevalence_without_wave12.csv",
                },
                {
                    "id": "02_participant_match_counts",
                    "source_table": "02_participant_match_counts.csv",
                    "source_columns": {
                        "speeddating.arff": ["wave", "match"],
                        "output/audit/v5/participant_assignments.csv": [
                            "row_index",
                            "participant_id",
                        ],
                    },
                    "variant": "02_participant_match_counts_without_wave12.png",
                    "variant_source_table": "02_participant_match_counts_without_wave12.csv",
                },
                {
                    "id": "03_core_rating_distributions",
                    "source_table": "03_core_rating_distributions.csv",
                    "source_columns": {"speeddating.arff": list(CORE_RATINGS)},
                },
                {
                    "id": "04_core_missing_outcome_rates",
                    "source_table": "04_core_missing_outcome_rates.csv",
                    "source_columns": {
                        "speeddating.arff": [*CORE_RATINGS, "match"],
                        "output/audit/v5/participant_assignments.csv": [
                            "row_index",
                            "core_rating_missing",
                        ],
                    },
                },
                {
                    "id": "05_early_wave_expectation_coverage",
                    "source_table": "05_early_wave_expectation_coverage.csv",
                    "source_columns": {
                        "speeddating.arff": [
                            "wave",
                            "expected_num_interested_in_me",
                            "match",
                        ]
                    },
                },
                {
                    "id": "06_reciprocal_pair_status",
                    "source_table": "06_reciprocal_pair_status.csv",
                    "source_columns": {
                        "output/audit/v5/reciprocal_pairs.csv": [
                            "row_a",
                            "row_b",
                            "pairing_status",
                        ],
                        "output/audit/v5/one_sided_rows.csv": ["row_index", "classification"],
                    },
                },
            ],
        }
        write_json(stage / "m2_manifest.json", manifest)
        return commit_staged_artifacts(stage, output_dir, ARTIFACT_NAMES)


def main() -> None:
    args = parse_args()
    paths = build_milestone_2(
        input_path=args.input,
        audit_dir=args.audit_dir,
        output_dir=args.output_dir,
        force=args.force,
    )
    print("wrote Milestone 2 artifacts:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
