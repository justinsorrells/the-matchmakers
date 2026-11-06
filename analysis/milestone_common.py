"""Shared, contract-enforcing utilities for Matchmakers milestones 2 and 3."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .data_audit import CORE_RATINGS, load_arff


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "speeddating.arff"
AUDIT_V5 = ROOT / "output" / "audit" / "v5"
M2_OUTPUT = ROOT / "output" / "m2"
M3_OUTPUT = ROOT / "output" / "m3"

PINNED_VERSIONS = {
    "python": "3.14.5",
    "matplotlib": "3.11.2",
    "numpy": "2.5.3",
    "pandas": "3.0.6",
    "scikit-learn": "1.9.1",
    "scipy": "1.18.1",
}


class ContractViolation(ValueError):
    """Raised when code or data would violate the frozen Version 5.2 contract."""


def validate_pinned_environment() -> dict[str, str]:
    """Return installed versions and fail unless the pinned environment is active."""
    observed = {
        "python": platform.python_version(),
        "matplotlib": version("matplotlib"),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit-learn": version("scikit-learn"),
        "scipy": version("scipy"),
    }
    mismatches = {
        name: {"expected": expected, "observed": observed[name]}
        for name, expected in PINNED_VERSIONS.items()
        if observed[name] != expected
    }
    if mismatches:
        raise ContractViolation(f"pinned environment mismatch: {mismatches}")
    return observed


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_feature_policy(audit_dir: Path = AUDIT_V5) -> dict[str, Any]:
    return read_json(audit_dir / "prohibited_features.json")


def validate_feature_allowlist(
    features: Sequence[str],
    allowlist_name: str,
    *,
    audit_dir: Path = AUDIT_V5,
    require_exact: bool = False,
) -> list[str]:
    """Fail if model features violate the named frozen allowlist or feature policy."""
    policy = load_feature_policy(audit_dir)
    allowlists = policy["feature_allowlists"]
    if allowlist_name not in allowlists:
        raise ContractViolation(f"unknown feature allowlist: {allowlist_name}")

    ordered = list(features)
    if len(ordered) != len(set(ordered)):
        raise ContractViolation(f"duplicate model features: {ordered}")

    allowed = list(allowlists[allowlist_name])
    outside = sorted(set(ordered) - set(allowed))
    prohibited = sorted(
        set(ordered) & set(policy["prohibited_predictive_features"])
    )
    categorical_bins = sorted(set(ordered) & set(policy["categorical_d_columns"]))
    if outside or prohibited or categorical_bins:
        raise ContractViolation(
            "invalid model feature set: "
            f"outside_allowlist={outside}, prohibited={prohibited}, "
            f"categorical_d_columns={categorical_bins}"
        )
    if require_exact and ordered != allowed:
        raise ContractViolation(
            f"{allowlist_name} feature order must equal the frozen allowlist; "
            f"expected={allowed}, observed={ordered}"
        )
    return ordered


def assert_held_out_wave_separation(
    train_waves: Iterable[int], test_waves: Iterable[int], held_out_wave: int
) -> None:
    train = {int(wave) for wave in train_waves}
    test = {int(wave) for wave in test_waves}
    if held_out_wave in train:
        raise ContractViolation(
            f"held-out wave {held_out_wave} appears in its own training data"
        )
    if test != {int(held_out_wave)}:
        raise ContractViolation(
            f"outer test data must contain only wave {held_out_wave}; observed={test}"
        )
    if train & test:
        raise ContractViolation(f"train/test wave overlap: {sorted(train & test)}")


def load_analysis_frame(
    input_path: Path = RAW_DATA, audit_dir: Path = AUDIT_V5
) -> pd.DataFrame:
    """Load the ARFF and attach only frozen participant identifiers and folds."""
    frame = load_arff(input_path).reset_index()
    assignments = pd.read_csv(audit_dir / "participant_assignments.csv")

    if frame["row_index"].duplicated().any() or assignments["row_index"].duplicated().any():
        raise ContractViolation("row_index must be unique in data and assignments")
    if set(frame["row_index"]) != set(assignments["row_index"]):
        raise ContractViolation("ARFF and participant assignment row indices differ")

    assignments = assignments.set_index("row_index").sort_index()
    frame = frame.set_index("row_index").sort_index()
    observed_wave = frame["wave"].astype(int)
    observed_match = frame["match"].astype(int)
    if not observed_wave.equals(assignments["wave"].astype(int)):
        raise ContractViolation("wave values differ from the frozen assignment artifact")
    if not observed_match.equals(assignments["match"].astype(int)):
        raise ContractViolation("match values differ from the frozen assignment artifact")

    frame["wave"] = observed_wave
    frame["match"] = observed_match
    frame["participant_id"] = assignments["participant_id"].astype(str)
    frame["actor_fold"] = assignments["actor_fold"].astype(int)
    frame["core_rating_missing"] = assignments["core_rating_missing"].astype(bool)

    computed_missing = frame[CORE_RATINGS].isna().any(axis=1)
    if not computed_missing.equals(frame["core_rating_missing"]):
        raise ContractViolation("core-missing mask differs from the frozen assignment artifact")
    return frame.reset_index()


def refuse_existing_artifacts(
    output_dir: Path, artifact_names: Sequence[str], force: bool
) -> None:
    existing = [output_dir / name for name in artifact_names if (output_dir / name).exists()]
    if existing and not force:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to replace existing milestone artifacts: {formatted}; use --force"
        )


def commit_staged_artifacts(
    staging_dir: Path, output_dir: Path, artifact_names: Sequence[str]
) -> list[Path]:
    """Move a complete staged artifact set into its output directory."""
    missing = [name for name in artifact_names if not (staging_dir / name).is_file()]
    if missing:
        raise ContractViolation(f"staged artifact set is incomplete: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in artifact_names:
        source = staging_dir / name
        target = output_dir / name
        shutil.copyfile(source, target)
        written.append(target)
    return written


def frozen_input_hashes(
    input_path: Path = RAW_DATA, audit_dir: Path = AUDIT_V5
) -> dict[str, str]:
    paths = [
        input_path,
        ROOT / "output" / "docx" / "The_Matchmakers_Project_Charter.docx",
        ROOT / "analysis" / "data_audit.py",
        *sorted(path for path in audit_dir.iterdir() if path.is_file()),
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}
