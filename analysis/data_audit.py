"""Reproduce and export the data contract for The Matchmakers project.

The default run is read-only: it loads ``speeddating.arff``, validates the
charter's expected values, and prints JSON. Pass ``--output-dir`` to write the
JSON audit, participant assignments, reciprocal pairs, per-fold couple
membership, one-sided rows, and the prohibited-feature list. Existing
artifacts are preserved unless ``--force`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import arff


CORE_RATINGS = ["attractive", "sincere", "intelligence", "funny", "ambition"]

CATEGORICAL_D_COLUMNS = [
    "d_d_age",
    "d_importance_same_race",
    "d_importance_same_religion",
    "d_pref_o_attractive",
    "d_pref_o_sincere",
    "d_pref_o_intelligence",
    "d_pref_o_funny",
    "d_pref_o_ambitious",
    "d_pref_o_shared_interests",
    "d_attractive_o",
    "d_sinsere_o",
    "d_intelligence_o",
    "d_funny_o",
    "d_ambitous_o",
    "d_shared_interests_o",
    "d_attractive_important",
    "d_sincere_important",
    "d_intellicence_important",
    "d_funny_important",
    "d_ambtition_important",
    "d_shared_interests_important",
    "d_attractive",
    "d_sincere",
    "d_intelligence",
    "d_funny",
    "d_ambition",
    "d_attractive_partner",
    "d_sincere_partner",
    "d_intelligence_partner",
    "d_funny_partner",
    "d_ambition_partner",
    "d_shared_interests_partner",
    "d_sports",
    "d_tvsports",
    "d_exercise",
    "d_dining",
    "d_museums",
    "d_art",
    "d_hiking",
    "d_gaming",
    "d_clubbing",
    "d_reading",
    "d_tv",
    "d_theater",
    "d_movies",
    "d_concerts",
    "d_music",
    "d_shopping",
    "d_yoga",
    "d_interests_correlate",
    "d_expected_happy_with_sd_people",
    "d_expected_num_interested_in_me",
    "d_expected_num_matches",
    "d_like",
    "d_guess_prob_liked",
]

COUPLE_FEATURES = [
    f"{gender}_{feature}"
    for gender in ("woman", "man")
    for feature in CORE_RATINGS
]

DERIVED_FEATURES = [
    "core_profile_missing",
    "woman_core_profile_missing",
    "man_core_profile_missing",
]

CONTEXT_FEATURES = [
    "gender",
    "age",
    "age_o",
    "d_age",
    "race",
    "race_o",
    "samerace",
    "importance_same_race",
    "importance_same_religion",
    "field",
    "interests_correlate",
]

FEATURE_ALLOWLISTS = {
    "core": CORE_RATINGS,
    "core_imputation_sensitivity": CORE_RATINGS + ["core_profile_missing"],
    "expectation_waves_1_to_5": CORE_RATINGS
    + ["expected_num_interested_in_me"],
    "couple_gender_ordered": COUPLE_FEATURES,
    "couple_imputation_sensitivity": COUPLE_FEATURES
    + ["woman_core_profile_missing", "man_core_profile_missing"],
    "context": CONTEXT_FEATURES,
}

EXPECTED_COMPLETE_CORE_COUPLE_MEMBERSHIP = {
    0: {"train": 2475, "test": 202, "dropped": 1402},
    1: {"train": 2547, "test": 180, "dropped": 1352},
    2: {"train": 2609, "test": 164, "dropped": 1306},
    3: {"train": 2692, "test": 145, "dropped": 1242},
    4: {"train": 2739, "test": 134, "dropped": 1206},
}

EXPECTED_ACTOR_FOLD_SHA256 = (
    "abeb7d4c6606e38611416044cedbdd92cc33cb9c56387ded5b4ab91915aa78dc"
)

PROHIBITED_PREDICTIVE_FEATURES = [
    "has_null",
    "expected_num_matches",
    "attractive_o",
    "sinsere_o",
    "intelligence_o",
    "funny_o",
    "ambitous_o",
    "shared_interests_o",
    "attractive_partner",
    "sincere_partner",
    "intelligence_partner",
    "funny_partner",
    "ambition_partner",
    "shared_interests_partner",
    "like",
    "guess_prob_liked",
    "met",
    "decision",
    "decision_o",
    "match",
] + CATEGORICAL_D_COLUMNS

PARTICIPANT_SIGNATURE_COLUMNS = [
    "wave",
    "gender",
    "age",
    "race",
    "importance_same_race",
    "importance_same_religion",
    "field",
    "attractive_important",
    "sincere_important",
    "intellicence_important",
    "funny_important",
    "ambtition_important",
    "shared_interests_important",
    "attractive",
    "sincere",
    "intelligence",
    "funny",
    "ambition",
    "sports",
    "tvsports",
    "exercise",
    "dining",
    "museums",
    "art",
    "hiking",
    "gaming",
    "clubbing",
    "reading",
    "tv",
    "theater",
    "movies",
    "concerts",
    "music",
    "shopping",
    "yoga",
    "expected_happy_with_sd_people",
    "expected_num_interested_in_me",
    "expected_num_matches",
]

# Post-date ratings are used here only to verify exact reciprocal rows. They are
# prohibited from predictive models.
STRICT_RECIPROCAL_COLUMNS = [
    "wave",
    "gender",
    "age",
    "age_o",
    "race",
    "race_o",
    "attractive_important",
    "sincere_important",
    "intellicence_important",
    "funny_important",
    "ambtition_important",
    "shared_interests_important",
    "pref_o_attractive",
    "pref_o_sincere",
    "pref_o_intelligence",
    "pref_o_funny",
    "pref_o_ambitious",
    "pref_o_shared_interests",
    "attractive_partner",
    "sincere_partner",
    "intelligence_partner",
    "funny_partner",
    "ambition_partner",
    "shared_interests_partner",
    "attractive_o",
    "sinsere_o",
    "intelligence_o",
    "funny_o",
    "ambitous_o",
    "shared_interests_o",
    "decision",
    "decision_o",
    "match",
]

LENIENT_RECIPROCAL_COLUMNS = STRICT_RECIPROCAL_COLUMNS[:18] + [
    "decision",
    "decision_o",
    "match",
]

POST_DATE_RATING_COLUMNS = [
    "attractive_partner",
    "sincere_partner",
    "intelligence_partner",
    "funny_partner",
    "ambition_partner",
    "shared_interests_partner",
    "attractive_o",
    "sinsere_o",
    "intelligence_o",
    "funny_o",
    "ambitous_o",
    "shared_interests_o",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("speeddating.arff"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write audit artifacts to this directory. Omit for a read-only run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing artifacts in --output-dir.",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Report observed values without checking the charter's expected values.",
    )
    return parser.parse_args()


def load_arff(path: Path) -> pd.DataFrame:
    records, _ = arff.loadarff(path)
    frame = pd.DataFrame(records)
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].str.decode("utf-8")
    for column in ("has_null", "samerace", "decision", "decision_o", "match"):
        frame[column] = pd.to_numeric(frame[column])
    frame.index.name = "row_index"
    return frame


def normalized(value: Any) -> Any:
    if pd.isna(value):
        return "<NA>"
    if hasattr(value, "item"):
        return value.item()
    return value


def fingerprint(row: pd.Series, columns: list[str]) -> tuple[Any, ...]:
    return tuple(normalized(row[column]) for column in columns)


def reciprocal_values(row: pd.Series, include_ratings: bool) -> tuple[Any, ...]:
    other_gender = "male" if row["gender"] == "female" else "female"
    values = [
        row["wave"],
        other_gender,
        row["age_o"],
        row["age"],
        row["race_o"],
        row["race"],
        row["pref_o_attractive"],
        row["pref_o_sincere"],
        row["pref_o_intelligence"],
        row["pref_o_funny"],
        row["pref_o_ambitious"],
        row["pref_o_shared_interests"],
        row["attractive_important"],
        row["sincere_important"],
        row["intellicence_important"],
        row["funny_important"],
        row["ambtition_important"],
        row["shared_interests_important"],
    ]
    if include_ratings:
        values.extend(
            [
                row["attractive_o"],
                row["sinsere_o"],
                row["intelligence_o"],
                row["funny_o"],
                row["ambitous_o"],
                row["shared_interests_o"],
                row["attractive_partner"],
                row["sincere_partner"],
                row["intelligence_partner"],
                row["funny_partner"],
                row["ambition_partner"],
                row["shared_interests_partner"],
            ]
        )
    values.extend([row["decision_o"], row["decision"], row["match"]])
    return tuple(normalized(value) for value in values)


def reciprocal_map(
    frame: pd.DataFrame,
    columns: list[str],
    include_ratings: bool,
    eligible_rows: set[int] | None = None,
) -> tuple[dict[int, int], Counter[int]]:
    index: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row_index, row in frame.iterrows():
        index[fingerprint(row, columns)].append(int(row_index))

    partners: dict[int, int] = {}
    candidate_counts: Counter[int] = Counter()
    for row_index, row in frame.iterrows():
        if eligible_rows is not None and row_index not in eligible_rows:
            continue
        candidates = [
            candidate
            for candidate in index.get(reciprocal_values(row, include_ratings), [])
            if candidate != row_index
        ]
        candidate_counts[len(candidates)] += 1
        if len(candidates) == 1:
            partners[int(row_index)] = candidates[0]
    return partners, candidate_counts


def symmetric_rows(partners: dict[int, int]) -> set[int]:
    return {
        row_index
        for row_index, partner_index in partners.items()
        if partners.get(partner_index) == row_index
    }


def outlier_note(frame: pd.DataFrame, row_indices: list[int]) -> str:
    notes: list[str] = []
    for row_index in row_indices:
        for column in POST_DATE_RATING_COLUMNS:
            value = frame.at[row_index, column]
            if pd.notna(value) and (value < 0 or value > 10):
                notes.append(f"row {row_index} {column}={value:g}")
    return "; ".join(notes)


def graph_component_count(pairs: pd.DataFrame, participant_count: int) -> int:
    parent = list(range(participant_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in pairs.itertuples(index=False):
        union(int(row.participant_a_number), int(row.participant_b_number))
    return len({find(value) for value in range(participant_count)})


def graph_components_by_wave(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    participant_numbers: pd.Series,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for wave in sorted(int(value) for value in frame["wave"].unique()):
        wave_rows = frame["wave"].eq(wave)
        nodes = sorted(set(int(value) for value in participant_numbers[wave_rows]))
        parent = {node: node for node in nodes}

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for row in pairs.loc[pairs["wave"].eq(wave)].itertuples(index=False):
            union(int(row.participant_a_number), int(row.participant_b_number))
        result[str(wave)] = len({find(node) for node in nodes})
    return result


def assign_actor_folds(
    frame: pd.DataFrame,
    participant_ids: pd.Series,
) -> dict[str, int]:
    participant_metadata = (
        pd.DataFrame(
            {
                "participant_id": participant_ids,
                "wave": frame["wave"].astype(int),
                "gender": frame["gender"],
            }
        )
        .drop_duplicates("participant_id")
        .sort_values(["wave", "gender", "participant_id"])
    )
    rng = np.random.default_rng(4622)
    fold_by_participant: dict[str, int] = {}
    for wave in sorted(participant_metadata["wave"].unique()):
        for gender in ["female", "male"]:
            participant_group = sorted(
                participant_metadata.loc[
                    participant_metadata["wave"].eq(wave)
                    & participant_metadata["gender"].eq(gender),
                    "participant_id",
                ].tolist()
            )
            for position, participant_id in enumerate(rng.permutation(participant_group)):
                fold_by_participant[str(participant_id)] = int(position % 5)
    return fold_by_participant


def actor_fold_sha256(actor_folds: dict[str, int]) -> str:
    canonical = "".join(
        f"{participant_id},{actor_folds[participant_id]}\n"
        for participant_id in sorted(actor_folds)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def couple_fold_membership_table(pairs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, int]] = []
    for fold in range(5):
        person_a_in_fold = pairs["fold_a"].eq(fold)
        person_b_in_fold = pairs["fold_b"].eq(fold)
        train = ~person_a_in_fold & ~person_b_in_fold
        test = person_a_in_fold & person_b_in_fold
        dropped = person_a_in_fold ^ person_b_in_fold
        core_complete = pairs["core_complete"]
        records.append(
            {
                "fold": fold,
                "all_train_pairs": int(train.sum()),
                "all_test_pairs": int(test.sum()),
                "all_dropped_pairs": int(dropped.sum()),
                "complete_core_train_pairs": int((train & core_complete).sum()),
                "complete_core_test_pairs": int((test & core_complete).sum()),
                "complete_core_dropped_pairs": int((dropped & core_complete).sum()),
                "complete_core_test_matches": int(
                    pairs.loc[test & core_complete, "match"].sum()
                ),
            }
        )
    return pd.DataFrame(records)


def participant_group_size_summary(
    frame: pd.DataFrame,
    participant_ids: pd.Series,
    unpaired: pd.DataFrame,
) -> dict[str, Any]:
    participant_rows = (
        pd.DataFrame(
            {
                "participant_id": participant_ids,
                "wave": frame["wave"].astype(int),
                "gender": frame["gender"],
            }
        )
        .groupby("participant_id", as_index=False)
        .agg(wave=("wave", "first"), gender=("gender", "first"), rows=("wave", "size"))
    )
    people_by_wave_gender = (
        participant_rows.groupby(["wave", "gender"])["participant_id"]
        .nunique()
        .to_dict()
    )
    one_sided_by_participant = unpaired.groupby("participant_id").size().to_dict()
    participant_rows["opposite_gender_participants"] = participant_rows.apply(
        lambda row: people_by_wave_gender[
            (row["wave"], "male" if row["gender"] == "female" else "female")
        ],
        axis=1,
    )
    participant_rows["one_sided_rows"] = (
        participant_rows["participant_id"]
        .map(one_sided_by_participant)
        .fillna(0)
        .astype(int)
    )
    participant_rows["expected_rows"] = (
        participant_rows["opposite_gender_participants"]
        + participant_rows["one_sided_rows"]
    )
    mismatches = participant_rows.loc[
        participant_rows["rows"].ne(participant_rows["expected_rows"])
    ]
    return {
        "participants_checked": int(len(participant_rows)),
        "participants_with_unexpected_group_size": int(len(mismatches)),
        "all_group_sizes_match_opposite_gender_plus_one_sided": bool(
            mismatches.empty
        ),
    }


def reciprocal_audit(
    frame: pd.DataFrame, participant_numbers: pd.Series
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    strict_partners, strict_candidate_counts = reciprocal_map(
        frame, STRICT_RECIPROCAL_COLUMNS, include_ratings=True
    )
    strict_rows = symmetric_rows(strict_partners)
    strict_unpaired = set(int(value) for value in frame.index) - strict_rows

    lenient_partners, _ = reciprocal_map(
        frame,
        LENIENT_RECIPROCAL_COLUMNS,
        include_ratings=False,
        eligible_rows=strict_unpaired,
    )
    recovered_rows = symmetric_rows(lenient_partners) & strict_unpaired
    one_sided_rows = strict_unpaired - recovered_rows

    pair_records: list[dict[str, Any]] = []
    used: set[int] = set()
    pair_number = 0
    for status, rows, partner_map in (
        ("strict", strict_rows, strict_partners),
        ("recovered_postdate_rating_outlier", recovered_rows, lenient_partners),
    ):
        for row_index in sorted(rows):
            partner_index = partner_map[row_index]
            if row_index in used or partner_index in used:
                continue
            used.update([row_index, partner_index])
            row_a, row_b = row_index, partner_index
            if frame.at[row_a, "gender"] == "male":
                row_a, row_b = row_b, row_a
            pair_number += 1
            pair_records.append(
                {
                    "pair_id": f"D{pair_number:05d}",
                    "wave": int(frame.at[row_a, "wave"]),
                    "row_a": int(row_a),
                    "row_b": int(row_b),
                    "participant_a": f"P{int(participant_numbers.at[row_a]) + 1:04d}",
                    "participant_b": f"P{int(participant_numbers.at[row_b]) + 1:04d}",
                    "participant_a_number": int(participant_numbers.at[row_a]),
                    "participant_b_number": int(participant_numbers.at[row_b]),
                    "gender_a": frame.at[row_a, "gender"],
                    "gender_b": frame.at[row_b, "gender"],
                    "match": int(frame.at[row_a, "match"]),
                    "pairing_status": status,
                    "data_note": outlier_note(frame, [row_a, row_b]),
                }
            )

    pairs = pd.DataFrame(pair_records)
    unpaired_columns = [
        "wave",
        "gender",
        "age",
        "age_o",
        "decision",
        "decision_o",
        "match",
    ]
    unpaired = frame.loc[sorted(one_sided_rows), unpaired_columns].reset_index()
    unpaired["participant_id"] = [
        f"P{int(participant_numbers.at[row_index]) + 1:04d}"
        for row_index in unpaired["row_index"]
    ]
    unpaired["classification"] = "one_sided_record"

    unordered_participant_pairs = pairs.apply(
        lambda row: tuple(sorted((row["participant_a"], row["participant_b"]))),
        axis=1,
    )
    label_agreement = all(
        frame.at[row.row_a, "match"] == frame.at[row.row_b, "match"]
        for row in pairs.itertuples(index=False)
    )
    recovered_frame = frame.loc[sorted(recovered_rows), POST_DATE_RATING_COLUMNS]
    recovered_outlier_values = {
        column: sorted(
            float(value)
            for value in recovered_frame[column].dropna()
            if value < 0 or value > 10
        )
        for column in POST_DATE_RATING_COLUMNS
        if any(
            value < 0 or value > 10
            for value in recovered_frame[column].dropna()
        )
    }

    summary = {
        "strict_paired_rows": len(strict_rows),
        "strict_pairs": int((pairs["pairing_status"] == "strict").sum()),
        "recovered_outlier_rows": len(recovered_rows),
        "recovered_outlier_pairs": int(
            (pairs["pairing_status"] == "recovered_postdate_rating_outlier").sum()
        ),
        "recovered_outlier_values": recovered_outlier_values,
        "verified_or_recovered_paired_rows": int(len(pairs) * 2),
        "verified_or_recovered_pairs": int(len(pairs)),
        "one_sided_rows": int(len(unpaired)),
        "mutual_match_pairs": int(pairs["match"].sum()),
        "match_label_agreement": bool(label_agreement),
        "strict_candidate_count_distribution": {
            str(key): int(value)
            for key, value in sorted(strict_candidate_counts.items())
        },
        "one_sided_rows_by_wave": {
            str(int(wave)): int(count)
            for wave, count in unpaired.groupby("wave").size().items()
        },
        "one_sided_all_partner_decisions_no": bool((unpaired["decision_o"] == 0).all()),
        "one_sided_all_nonmatches": bool((unpaired["match"] == 0).all()),
        "repeated_unordered_participant_pairs": int(
            unordered_participant_pairs.duplicated().sum()
        ),
        "participant_partner_graph_components": graph_component_count(
            pairs, int(participant_numbers.nunique())
        ),
    }
    return summary, pairs, unpaired


def build_audit(
    frame: pd.DataFrame,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    participant_numbers = frame.groupby(
        PARTICIPANT_SIGNATURE_COLUMNS, dropna=False, sort=False
    ).ngroup()
    participant_numbers = pd.Series(participant_numbers, index=frame.index)
    participant_sizes = participant_numbers.value_counts()
    participant_ids = participant_numbers.map(lambda value: f"P{int(value) + 1:04d}")
    actor_folds = assign_actor_folds(frame, participant_ids)

    reciprocal_summary, pairs, unpaired = reciprocal_audit(frame, participant_numbers)

    positive_rows = int(frame["match"].sum())
    prevalence = float(frame["match"].mean())
    wave_rates = frame.groupby("wave")["match"].mean()
    wave_positive_rows = frame.groupby("wave")["match"].sum()
    correlations = {
        column: float(frame[column].corr(frame["match"])) for column in CORE_RATINGS
    }
    core_missing_rows = frame[CORE_RATINGS].isna().any(axis=1)
    core_complete_positive_rows_by_wave = (
        frame.loc[~core_missing_rows].groupby("wave")["match"].sum()
    )
    early_waves = frame["wave"].le(5)
    expectation_missing_early = early_waves & frame[
        "expected_num_interested_in_me"
    ].isna()
    expectation_observed_early = early_waves & frame[
        "expected_num_interested_in_me"
    ].notna()
    expectation_observed = frame.groupby("wave")["expected_num_interested_in_me"].apply(
        lambda values: int(values.notna().sum())
    )
    participant_binomial = (
        frame.assign(participant_number=participant_numbers)
        .groupby(["participant_number", "wave"], as_index=False)
        .agg(successes=("match", "sum"), trials=("match", "size"))
    )
    wave_prevalence = frame.groupby("wave")["match"].mean()
    participant_binomial["wave_prevalence"] = participant_binomial["wave"].map(
        wave_prevalence
    )
    pearson_chi_square = (
        (
            participant_binomial["successes"]
            - participant_binomial["trials"]
            * participant_binomial["wave_prevalence"]
        )
        ** 2
        / (
            participant_binomial["trials"]
            * participant_binomial["wave_prevalence"]
            * (1 - participant_binomial["wave_prevalence"])
        )
    ).sum()
    participant_dispersion_df = len(participant_binomial) - frame["wave"].nunique()

    pairs["core_complete"] = [
        bool(
            not core_missing_rows.at[int(row.row_a)]
            and not core_missing_rows.at[int(row.row_b)]
        )
        for row in pairs.itertuples(index=False)
    ]
    pairs["fold_a"] = pairs["participant_a"].map(actor_folds).astype(int)
    pairs["fold_b"] = pairs["participant_b"].map(actor_folds).astype(int)
    pairs["couple_fold"] = [
        str(row.fold_a)
        if row.fold_a == row.fold_b
        else "mixed"
        for row in pairs.itertuples(index=False)
    ]
    couple_membership = couple_fold_membership_table(pairs)
    pure_couple_counts = (
        pairs.loc[pairs["couple_fold"].ne("mixed")]
        .groupby(["wave", "couple_fold"])
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [sorted(int(value) for value in frame["wave"].unique()), [str(i) for i in range(5)]],
                names=["wave", "couple_fold"],
            ),
            fill_value=0,
        )
    )
    complete_core_pure_couple_counts = (
        pairs.loc[pairs["core_complete"] & pairs["couple_fold"].ne("mixed")]
        .groupby(["wave", "couple_fold"])
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [
                    sorted(int(value) for value in frame["wave"].unique()),
                    [str(i) for i in range(5)],
                ],
                names=["wave", "couple_fold"],
            ),
            fill_value=0,
        )
    )
    complete_core_empty_cells = [
        {"wave": int(wave), "fold": int(fold)}
        for wave, fold in complete_core_pure_couple_counts[
            complete_core_pure_couple_counts.eq(0)
        ].index
    ]
    reciprocal_summary["participant_partner_graph_components_by_wave"] = (
        graph_components_by_wave(frame, pairs, participant_numbers)
    )
    reciprocal_summary["core_complete_pairs"] = int(pairs["core_complete"].sum())
    reciprocal_summary["core_missing_pairs"] = int((~pairs["core_complete"]).sum())
    reciprocal_summary["core_missing_pair_matches"] = int(
        pairs.loc[~pairs["core_complete"], "match"].sum()
    )

    categorical_d_columns_observed = sorted(
        column
        for column in frame.columns
        if column.startswith("d_") and column != "d_age"
    )
    late_wave_rows = frame["wave"].ge(6)
    early_expectation_rows_are_core_complete = bool(
        (~core_missing_rows[expectation_observed_early]).all()
    )
    feature_policy_intersections = {
        name: sorted(set(features).intersection(PROHIBITED_PREDICTIVE_FEATURES))
        for name, features in FEATURE_ALLOWLISTS.items()
    }
    allowlist_categorical_d_columns = {
        name: sorted(set(features).intersection(CATEGORICAL_D_COLUMNS))
        for name, features in FEATURE_ALLOWLISTS.items()
    }
    group_size_summary = participant_group_size_summary(
        frame, participant_ids, unpaired
    )

    participant_artifact = pd.DataFrame(
        {
            "row_index": frame.index,
            "wave": frame["wave"].astype(int),
            "participant_id": participant_ids,
            "gender": frame["gender"],
            "actor_fold": participant_ids.map(actor_folds).astype(int),
            "match": frame["match"].astype(int),
            "core_rating_missing": core_missing_rows,
        }
    )

    audit = {
        "dataset": {
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "waves": int(frame["wave"].nunique()),
        },
        "outcome": {
            "positive_rows": positive_rows,
            "negative_rows": int(len(frame) - positive_rows),
            "positive_row_prevalence": prevalence,
            "wave_rate_min": float(wave_rates.min()),
            "wave_rate_min_wave": int(wave_rates.idxmin()),
            "wave_rate_max": float(wave_rates.max()),
            "wave_rate_max_wave": int(wave_rates.idxmax()),
            "positive_rows_by_wave": {
                str(int(wave)): int(count)
                for wave, count in wave_positive_rows.items()
            },
            "small_wave_positive_rows_all": {
                str(wave): int(wave_positive_rows.loc[wave])
                for wave in (6, 18, 20)
            },
            "small_wave_positive_rows_core_complete": {
                str(wave): int(core_complete_positive_rows_by_wave.loc[wave])
                for wave in (6, 18, 20)
            },
            "all_positive_f1": float(2 * prevalence / (1 + prevalence)),
            "majority_negative_f1": 0.0,
        },
        "participants": {
            "signature_groups": int(participant_sizes.size),
            "complete_core_participants": int(
                participant_ids.loc[~core_missing_rows].nunique()
            ),
            "imputation_sensitivity_participants": int(participant_ids.nunique()),
            "rows_per_group_mean": float(participant_sizes.mean()),
            "rows_per_group_min": int(participant_sizes.min()),
            "rows_per_group_max": int(participant_sizes.max()),
            "groups_with_missing_core_rating": int(
                participant_ids[core_missing_rows].nunique()
            ),
            "core_missing_rows": int(core_missing_rows.sum()),
            "core_missing_positive_rows": int(
                frame.loc[core_missing_rows, "match"].sum()
            ),
            "core_complete_rows": int((~core_missing_rows).sum()),
            "core_complete_positive_rows": int(
                frame.loc[~core_missing_rows, "match"].sum()
            ),
            "core_missing_group_match_rate": float(
                frame.loc[core_missing_rows, "match"].mean()
            ),
            "core_complete_group_match_rate": float(
                frame.loc[~core_missing_rows, "match"].mean()
            ),
            "pearson_dispersion_wave_fixed_effects": float(
                pearson_chi_square / participant_dispersion_df
            ),
            "pearson_dispersion_degrees_of_freedom": int(
                participant_dispersion_df
            ),
            "group_size_check": group_size_summary,
        },
        "missingness": {
            "rows_with_any_missing_value": int(frame.isna().any(axis=1).sum()),
            "rows_with_any_missing_value_rate": float(frame.isna().any(axis=1).mean()),
            "rows_with_any_missing_excluding_expected_interested": int(
                frame.drop(columns=["expected_num_interested_in_me"])
                .isna()
                .any(axis=1)
                .sum()
            ),
            "rows_with_any_missing_excluding_expected_interested_rate": float(
                frame.drop(columns=["expected_num_interested_in_me"])
                .isna()
                .any(axis=1)
                .mean()
            ),
            "core_rating_missing_rows": {
                column: int(frame[column].isna().sum()) for column in CORE_RATINGS
            },
            "expected_num_interested_in_me_missing_rows": int(
                frame["expected_num_interested_in_me"].isna().sum()
            ),
            "expected_num_matches_missing_rows": int(
                frame["expected_num_matches"].isna().sum()
            ),
            "expected_interested_observed_by_wave": {
                str(int(wave)): int(count)
                for wave, count in expectation_observed.items()
            },
            "early_wave_expected_interested_missing_rows": int(
                expectation_missing_early.sum()
            ),
            "early_wave_expected_interested_missing_participants": int(
                participant_ids[expectation_missing_early].nunique()
            ),
            "early_wave_expected_interested_missing_match_rate": float(
                frame.loc[expectation_missing_early, "match"].mean()
            ),
            "early_wave_expected_interested_missing_positive_rows": int(
                frame.loc[expectation_missing_early, "match"].sum()
            ),
            "early_wave_expected_interested_observed_rows": int(
                expectation_observed_early.sum()
            ),
            "early_wave_expected_interested_observed_positive_rows": int(
                frame.loc[expectation_observed_early, "match"].sum()
            ),
            "early_wave_expected_interested_observed_match_rate": float(
                frame.loc[expectation_observed_early, "match"].mean()
            ),
            "early_wave_expectation_rows_all_core_complete": (
                early_expectation_rows_are_core_complete
            ),
        },
        "categorical_bins": {
            "categorical_d_column_count": int(len(categorical_d_columns_observed)),
            "categorical_d_columns": categorical_d_columns_observed,
            "late_wave_rows": int(late_wave_rows.sum()),
            "late_wave_d_expected_num_interested_in_me_all_lowest_bin": bool(
                frame.loc[
                    late_wave_rows, "d_expected_num_interested_in_me"
                ].eq("[0-3]").all()
            ),
            "core_missing_rows": int(core_missing_rows.sum()),
            "core_missing_d_attractive_all_lowest_bin": bool(
                frame.loc[core_missing_rows, "d_attractive"].eq("[0-5]").all()
            ),
        },
        "fold_assignment": {
            "seed": 4622,
            "algorithm": "one default_rng; wave ascending; gender female then male; sorted participant IDs; permute; position modulo 5",
            "actor_folds": 5,
            "actor_fold_sha256": actor_fold_sha256(actor_folds),
            "actor_fold_participant_counts": {
                str(fold): int(count)
                for fold, count in participant_artifact.drop_duplicates(
                    "participant_id"
                )["actor_fold"].value_counts().sort_index().items()
            },
            "couple_fold_definition": "test fold if both people share it, else mixed",
            "pure_couple_counts_by_wave_fold": {
                f"{int(wave)}:{fold}": int(count)
                for (wave, fold), count in pure_couple_counts.items()
            },
            "wave_fold_cells": int(len(pure_couple_counts)),
            "wave_fold_cells_without_pure_couple": int(
                pure_couple_counts.eq(0).sum()
            ),
            "minimum_pure_couples_in_wave_fold": int(pure_couple_counts.min()),
            "pure_couples": int(pairs["couple_fold"].ne("mixed").sum()),
            "mixed_couples": int(pairs["couple_fold"].eq("mixed").sum()),
            "complete_core_pure_couple_counts_by_wave_fold": {
                f"{int(wave)}:{fold}": int(count)
                for (wave, fold), count in complete_core_pure_couple_counts.items()
            },
            "complete_core_wave_fold_cells": int(
                len(complete_core_pure_couple_counts)
            ),
            "complete_core_wave_fold_cells_with_pure_couple": int(
                complete_core_pure_couple_counts.gt(0).sum()
            ),
            "complete_core_wave_fold_cells_without_pure_couple": int(
                complete_core_pure_couple_counts.eq(0).sum()
            ),
            "complete_core_empty_wave_fold_cells": complete_core_empty_cells,
            "pure_complete_core_test_couples": int(
                complete_core_pure_couple_counts.sum()
            ),
            "pure_complete_core_test_matches": int(
                pairs.loc[
                    pairs["core_complete"] & pairs["couple_fold"].ne("mixed"),
                    "match",
                ].sum()
            ),
            "complete_core_outer_membership_by_fold": [
                {
                    "fold": int(row.fold),
                    "train": int(row.complete_core_train_pairs),
                    "test": int(row.complete_core_test_pairs),
                    "dropped": int(row.complete_core_dropped_pairs),
                }
                for row in couple_membership.itertuples(index=False)
            ],
        },
        "core_rating_medians": {
            column: float(value) for column, value in frame[CORE_RATINGS].median().items()
        },
        "core_rating_correlations_with_match": correlations,
        "largest_absolute_core_correlation": float(
            max(abs(value) for value in correlations.values())
        ),
        "reciprocal_rows": reciprocal_summary,
        "predictive_feature_policy": {
            "core_features": CORE_RATINGS,
            "feature_allowlists": FEATURE_ALLOWLISTS,
            "derived_features": DERIVED_FEATURES,
            "prohibited_features": PROHIBITED_PREDICTIVE_FEATURES,
            "prohibited_features_present_in_dataset": sorted(
                set(PROHIBITED_PREDICTIVE_FEATURES).intersection(frame.columns)
            ),
            "allowlist_prohibited_intersections": feature_policy_intersections,
            "allowlist_categorical_d_intersections": allowlist_categorical_d_columns,
        },
        "cross_platform_reproducibility": {
            "data_audit_json_float_comparison": "compare floating-point values with tolerance because correlations can differ in the last digits across platforms",
            "byte_identical_artifacts_except_data_audit_json": [
                "participant_assignments.csv",
                "reciprocal_pairs.csv",
                "couple_fold_membership.csv",
                "one_sided_rows.csv",
                "prohibited_features.json",
            ],
        },
    }
    return audit, participant_artifact, pairs, couple_membership, unpaired


def assert_close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"expected {expected}, observed {actual}")


def validate_expected(audit: dict[str, Any]) -> None:
    assert audit["dataset"] == {"rows": 8378, "columns": 123, "waves": 21}
    assert audit["outcome"]["positive_rows"] == 1380
    assert audit["outcome"]["negative_rows"] == 6998
    assert_close(audit["outcome"]["positive_row_prevalence"], 1380 / 8378)
    assert_close(audit["outcome"]["wave_rate_min"], 1 / 12)
    assert_close(audit["outcome"]["wave_rate_max"], 0.31)
    assert_close(audit["outcome"]["all_positive_f1"], 0.2828448452551752)
    assert audit["outcome"]["small_wave_positive_rows_all"] == {
        "6": 10,
        "18": 6,
        "20": 10,
    }
    assert audit["outcome"]["small_wave_positive_rows_core_complete"] == {
        "6": 9,
        "18": 6,
        "20": 10,
    }
    assert audit["participants"]["signature_groups"] == 551
    assert audit["participants"]["complete_core_participants"] == 542
    assert audit["participants"]["imputation_sensitivity_participants"] == 551
    assert_close(audit["participants"]["rows_per_group_mean"], 8378 / 551)
    assert audit["participants"]["rows_per_group_min"] == 5
    assert audit["participants"]["rows_per_group_max"] == 22
    assert audit["participants"]["groups_with_missing_core_rating"] == 9
    assert audit["participants"]["core_missing_rows"] == 105
    assert audit["participants"]["core_missing_positive_rows"] == 29
    assert audit["participants"]["core_complete_rows"] == 8273
    assert audit["participants"]["core_complete_positive_rows"] == 1351
    assert_close(
        audit["participants"]["core_missing_group_match_rate"], 29 / 105
    )
    assert_close(
        audit["participants"]["core_complete_group_match_rate"], 1351 / 8273
    )
    assert_close(
        audit["participants"]["pearson_dispersion_wave_fixed_effects"],
        2.0648527443409996,
    )
    assert (
        audit["participants"]["pearson_dispersion_degrees_of_freedom"] == 530
    )
    assert audit["participants"]["group_size_check"] == {
        "participants_checked": 551,
        "participants_with_unexpected_group_size": 0,
        "all_group_sizes_match_opposite_gender_plus_one_sided": True,
    }
    missingness = audit["missingness"]
    assert missingness["rows_with_any_missing_value"] == 7330
    assert missingness["rows_with_any_missing_excluding_expected_interested"] == 3519
    assert set(missingness["core_rating_missing_rows"].values()) == {105}
    assert missingness["expected_num_interested_in_me_missing_rows"] == 6578
    assert missingness["expected_num_matches_missing_rows"] == 1173
    assert missingness["early_wave_expected_interested_missing_rows"] == 46
    assert missingness["early_wave_expected_interested_missing_participants"] == 4
    assert missingness["early_wave_expected_interested_missing_positive_rows"] == 4
    assert missingness["early_wave_expected_interested_observed_rows"] == 1800
    assert (
        missingness["early_wave_expected_interested_observed_positive_rows"] == 330
    )
    assert_close(
        missingness["early_wave_expected_interested_missing_match_rate"], 4 / 46
    )
    assert_close(
        missingness["early_wave_expected_interested_observed_match_rate"], 330 / 1800
    )
    assert missingness["early_wave_expectation_rows_all_core_complete"] is True
    observed_waves = {
        int(wave)
        for wave, count in missingness["expected_interested_observed_by_wave"].items()
        if count > 0
    }
    assert observed_waves == {1, 2, 3, 4, 5}
    assert audit["core_rating_medians"] == {
        "attractive": 7.0,
        "sincere": 8.0,
        "intelligence": 8.0,
        "funny": 8.0,
        "ambition": 8.0,
    }
    assert_close(audit["largest_absolute_core_correlation"], 0.0511, tolerance=1e-3)
    reciprocal = audit["reciprocal_rows"]
    assert reciprocal["strict_paired_rows"] == 8364
    assert reciprocal["strict_pairs"] == 4182
    assert reciprocal["recovered_outlier_rows"] == 4
    assert reciprocal["recovered_outlier_pairs"] == 2
    assert reciprocal["recovered_outlier_values"] == {
        "attractive_o": [10.5],
        "funny_o": [11.0],
    }
    assert reciprocal["verified_or_recovered_pairs"] == 4184
    assert reciprocal["one_sided_rows"] == 10
    assert reciprocal["one_sided_rows_by_wave"] == {"5": 10}
    assert reciprocal["mutual_match_pairs"] == 690
    assert reciprocal["match_label_agreement"] is True
    assert reciprocal["one_sided_all_partner_decisions_no"] is True
    assert reciprocal["one_sided_all_nonmatches"] is True
    assert reciprocal["repeated_unordered_participant_pairs"] == 0
    assert reciprocal["participant_partner_graph_components"] == 21
    assert reciprocal["participant_partner_graph_components_by_wave"] == {
        str(wave): 1 for wave in range(1, 22)
    }
    assert reciprocal["core_complete_pairs"] == 4079
    assert reciprocal["core_missing_pairs"] == 105
    assert reciprocal["core_missing_pair_matches"] == 29

    categorical_bins = audit["categorical_bins"]
    assert categorical_bins["categorical_d_column_count"] == 55
    assert categorical_bins["categorical_d_columns"] == sorted(
        CATEGORICAL_D_COLUMNS
    )
    assert categorical_bins["late_wave_rows"] == 6532
    assert (
        categorical_bins[
            "late_wave_d_expected_num_interested_in_me_all_lowest_bin"
        ]
        is True
    )
    assert categorical_bins["core_missing_rows"] == 105
    assert categorical_bins["core_missing_d_attractive_all_lowest_bin"] is True

    fold_assignment = audit["fold_assignment"]
    assert fold_assignment["seed"] == 4622
    assert fold_assignment["actor_folds"] == 5
    assert fold_assignment["actor_fold_sha256"] == EXPECTED_ACTOR_FOLD_SHA256
    assert fold_assignment["actor_fold_participant_counts"] == {
        "0": 123,
        "1": 114,
        "2": 111,
        "3": 106,
        "4": 97,
    }
    assert fold_assignment["couple_fold_definition"] == (
        "test fold if both people share it, else mixed"
    )
    assert fold_assignment["wave_fold_cells"] == 105
    assert fold_assignment["wave_fold_cells_without_pure_couple"] == 0
    assert fold_assignment["minimum_pure_couples_in_wave_fold"] >= 1
    assert fold_assignment["pure_couples"] == 845
    assert fold_assignment["mixed_couples"] == 3339
    assert fold_assignment["complete_core_wave_fold_cells"] == 105
    assert fold_assignment["complete_core_wave_fold_cells_with_pure_couple"] == 103
    assert fold_assignment["complete_core_wave_fold_cells_without_pure_couple"] == 2
    assert fold_assignment["complete_core_empty_wave_fold_cells"] == [
        {"wave": 6, "fold": 0},
        {"wave": 16, "fold": 3},
    ]
    assert fold_assignment["pure_complete_core_test_couples"] == 825
    assert fold_assignment["pure_complete_core_test_matches"] == 132
    assert fold_assignment["complete_core_outer_membership_by_fold"] == [
        {"fold": fold, **counts}
        for fold, counts in EXPECTED_COMPLETE_CORE_COUPLE_MEMBERSHIP.items()
    ]

    feature_policy = audit["predictive_feature_policy"]
    assert feature_policy["feature_allowlists"] == FEATURE_ALLOWLISTS
    assert feature_policy["derived_features"] == DERIVED_FEATURES
    assert all(
        not values
        for values in feature_policy["allowlist_prohibited_intersections"].values()
    )
    assert all(
        not values
        for values in feature_policy[
            "allowlist_categorical_d_intersections"
        ].values()
    )
    sensitivity_allowlists = {
        "core_imputation_sensitivity",
        "couple_imputation_sensitivity",
    }
    assert all(
        not (set(features) & set(DERIVED_FEATURES))
        for name, features in FEATURE_ALLOWLISTS.items()
        if name not in sensitivity_allowlists
    )
    assert set(FEATURE_ALLOWLISTS["core_imputation_sensitivity"]) & set(
        DERIVED_FEATURES
    ) == {"core_profile_missing"}
    assert set(FEATURE_ALLOWLISTS["couple_imputation_sensitivity"]) & set(
        DERIVED_FEATURES
    ) == {
        "woman_core_profile_missing",
        "man_core_profile_missing",
    }


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "audit": output_dir / "data_audit.json",
        "participants": output_dir / "participant_assignments.csv",
        "pairs": output_dir / "reciprocal_pairs.csv",
        "couple_membership": output_dir / "couple_fold_membership.csv",
        "unpaired": output_dir / "one_sided_rows.csv",
        "prohibited": output_dir / "prohibited_features.json",
    }


def write_artifacts(
    output_dir: Path,
    force: bool,
    audit: dict[str, Any],
    participants: pd.DataFrame,
    pairs: pd.DataFrame,
    couple_membership: pd.DataFrame,
    unpaired: pd.DataFrame,
) -> dict[str, Path]:
    paths = artifact_paths(output_dir)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to replace existing audit artifacts: {formatted}; use --force"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths["audit"].write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    participants.to_csv(paths["participants"], index=False)
    pairs.drop(columns=["participant_a_number", "participant_b_number"]).to_csv(
        paths["pairs"], index=False
    )
    couple_membership.to_csv(paths["couple_membership"], index=False)
    unpaired.to_csv(paths["unpaired"], index=False)
    paths["prohibited"].write_text(
        json.dumps(
            {
                "feature_allowlists": FEATURE_ALLOWLISTS,
                "derived_features": DERIVED_FEATURES,
                "categorical_d_columns": CATEGORICAL_D_COLUMNS,
                "numeric_d_age_allowed": True,
                "prohibited_predictive_features": PROHIBITED_PREDICTIVE_FEATURES,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return paths


def main() -> None:
    args = parse_args()
    frame = load_arff(args.input)
    audit, participants, pairs, couple_membership, unpaired = build_audit(frame)
    if not args.skip_checks:
        validate_expected(audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.output_dir is not None:
        paths = write_artifacts(
            args.output_dir,
            args.force,
            audit,
            participants,
            pairs,
            couple_membership,
            unpaired,
        )
        print("wrote artifacts:")
        for path in paths.values():
            print(f"  {path}")


if __name__ == "__main__":
    main()
