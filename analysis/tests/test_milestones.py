from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.data_audit import CORE_RATINGS
from analysis.m2_eda import ARTIFACT_NAMES as M2_ARTIFACTS
from analysis.m3_actor_baseline import (
    C_GRID,
    build_logistic_pipeline,
    choose_f1_threshold,
    select_logistic_c,
    validate_actor_primary_rows,
)
from analysis.milestone_common import (
    AUDIT_V5,
    ContractViolation,
    assert_held_out_wave_separation,
    load_analysis_frame,
    refuse_existing_artifacts,
    validate_feature_allowlist,
    validate_pinned_environment,
)


class FeatureContractTests(unittest.TestCase):
    def test_pinned_environment(self) -> None:
        observed = validate_pinned_environment()
        self.assertEqual(observed["python"], "3.14.5")
        self.assertEqual(observed["scikit-learn"], "1.9.1")
        self.assertEqual(observed["matplotlib"], "3.11.2")

    def test_core_allowlist_is_exact(self) -> None:
        self.assertEqual(
            validate_feature_allowlist(
                CORE_RATINGS, "core", audit_dir=AUDIT_V5, require_exact=True
            ),
            CORE_RATINGS,
        )

    def test_prohibited_feature_is_rejected(self) -> None:
        with self.assertRaises(ContractViolation):
            validate_feature_allowlist(
                [*CORE_RATINGS, "match"], "core", audit_dir=AUDIT_V5
            )

    def test_categorical_d_feature_is_rejected(self) -> None:
        with self.assertRaises(ContractViolation):
            validate_feature_allowlist(
                [*CORE_RATINGS, "d_attractive"], "core", audit_dir=AUDIT_V5
            )

    def test_outside_allowlist_feature_is_rejected(self) -> None:
        with self.assertRaises(ContractViolation):
            validate_feature_allowlist(
                [*CORE_RATINGS, "age"], "core", audit_dir=AUDIT_V5
            )

    def test_pipeline_contract(self) -> None:
        pipeline = build_logistic_pipeline(C_GRID[0], CORE_RATINGS)
        self.assertEqual(list(pipeline.named_steps), ["standard_scaler", "logistic_regression"])
        estimator = pipeline.named_steps["logistic_regression"]
        self.assertEqual(estimator.l1_ratio, 0.0)
        self.assertEqual(estimator.solver, "lbfgs")
        self.assertIsNone(estimator.class_weight)


class SplitContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load_analysis_frame()

    def test_actual_outer_waves_are_separated(self) -> None:
        core = validate_actor_primary_rows(self.frame, AUDIT_V5)
        for wave in sorted(core["wave"].unique()):
            assert_held_out_wave_separation(
                core.loc[core["wave"] != wave, "wave"],
                core.loc[core["wave"] == wave, "wave"],
                int(wave),
            )

    def test_held_out_wave_in_training_fails(self) -> None:
        with self.assertRaises(ContractViolation):
            assert_held_out_wave_separation([1, 2, 3], [3], 3)

    def test_wrong_test_wave_fails(self) -> None:
        with self.assertRaises(ContractViolation):
            assert_held_out_wave_separation([1, 2], [3, 4], 3)


class OutputAndMetricContractTests(unittest.TestCase):
    def test_m2_has_six_concepts_and_only_two_variants(self) -> None:
        primary_pngs = [
            name
            for name in M2_ARTIFACTS
            if name.endswith(".png") and "without_wave12" not in name
        ]
        variants = [name for name in M2_ARTIFACTS if name.endswith("without_wave12.png")]
        self.assertEqual(len(primary_pngs), 6)
        self.assertEqual(len(variants), 2)

    def test_existing_artifact_refuses_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "result.csv").write_text("already here\n")
            with self.assertRaises(FileExistsError):
                refuse_existing_artifacts(output, ["result.csv"], force=False)
            refuse_existing_artifacts(output, ["result.csv"], force=True)

    def test_f1_threshold_prefers_highest_exact_tie(self) -> None:
        y = np.array([1, 0, 1, 0], dtype=int)
        scores = np.array([0.9, 0.8, 0.7, 0.1], dtype=float)
        threshold, best_f1 = choose_f1_threshold(y, scores)
        candidates = []
        for value in np.unique(scores):
            score = float(
                2
                * np.sum((scores >= value) & (y == 1))
                / (
                    2 * np.sum((scores >= value) & (y == 1))
                    + np.sum((scores >= value) & (y == 0))
                    + np.sum((scores < value) & (y == 1))
                )
            )
            if score == best_f1:
                candidates.append(float(value))
        self.assertEqual(threshold, max(candidates))

    def test_logistic_c_prefers_smaller_value_on_exact_tie(self) -> None:
        scores = {0.01: 0.51, 0.1: 0.52, 1.0: 0.53, 10.0: 0.53}
        self.assertEqual(select_logistic_c(scores), 1.0)


if __name__ == "__main__":
    unittest.main()
