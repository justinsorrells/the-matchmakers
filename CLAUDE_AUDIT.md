# Claude Adversarial Re Audit Handoff for The Matchmakers Charter V5.2

## Request

Perform a read-only adversarial audit of Version 5.2. The sixth audit approved Version 5.1 for Milestone 3 after verifying the tie rule, expectation rows, small-wave sensitivity, fold-hash serialization, couple interval, context-field handling, derived-feature assertions, and unchanged artifacts. Version 5.2 resolves the one-sided-row wording conflict before Milestone 4 and adds two approval-gated context clarifications. It preserves every Version 3 through Version 5.1 decision and every exported fold. Do not edit files, fit models, use `--force`, run with an output directory, or modify `analysis/data_audit.py`, `speeddating.arff`, `output/audit/v4`, or `output/audit/v5`.

Determine whether the charter and data contract are implementable without unstated choices. Report defects in descending severity and separate them from grader decisions and optional improvements.

## Authoritative artifacts

- `output/docx/The_Matchmakers_Project_Charter.docx`
- `analysis/data_audit.py`
- `requirements.txt`
- `requirements.lock`
- `output/audit/v5/data_audit.json`
- `output/audit/v5/participant_assignments.csv`
- `output/audit/v5/reciprocal_pairs.csv`
- `output/audit/v5/couple_fold_membership.csv`
- `output/audit/v5/one_sided_rows.csv`
- `output/audit/v5/prohibited_features.json`

V4 remains unchanged under `output/audit/v4/`. Files under `tmp/` are builder and render working files, not analytical deliverables.

## Version 5.2 erratum change map

Every one-sided-row passage must contain this exact scope rule:

> Rerun the actor primary (including its average-precision and probability metrics) and the participant metrics without the ten one-sided rows. The new-actor, couple, expectation-subset, and Milestone 4 tree/forest analyses are not rerun.

| Sixth-audit change | Exact charter passage after Version 5.2 |
|---|---|
| Status and erratum note | Status: `Version 5.2 (erratum)`. The `Erratum 5.2` note says the sixth audit approved Version 5.1 for Milestone 3 and identifies the scope and context-input changes before Milestone 4. |
| Data and Initial Exploration, `Row pairing` row | `Include the ten rows as observed nonmatches in actor and participant analyses; exclude them from couple models.` followed by the exact scope rule above. |
| Preprocessing and Leakage Controls, expectation bullet | `All ten one-sided wave 5 rows are always included in those 1,800 rows.` followed by the exact scope rule above. |
| Preprocessing and Leakage Controls, one-sided-row bullet | `Include the ten one-sided wave 5 rows as observed nonmatches in actor and participant analyses. The couple primary uses 4,079 complete-core reciprocal pairs and excludes all ten one-sided rows.` followed by the exact scope rule above. |
| Risk Register, `One-sided records` response | `Treat the ten as observed failures for actor and participant analyses and exclude them from pair models.` followed by the exact scope rule above. |
| Acceptance Criterion 6 | The criterion says the expectation comparison always uses the identical 1,800 rows, including all ten one-sided wave 5 rows, and then states the exact scope rule above. |
| Acceptance Criterion 8 | The imputation requirements are followed by the exact scope rule above. |
| Preprocessing and Leakage Controls, context bullet | `If approved, use the same 8,273 complete-core rows as the actor primary, LOWO, and the L2 logistic pipeline only.` It also says: `The loader represents missing nominal values as the string '?'; treat '?' as the explicit missing level for gender, race, race_o, and field.` |

## Version 5.1 erratum change map

| Fifth-audit requirement | Charter section | Data contract evidence |
|---|---|---|
| Every logistic tuning step resolves exact ties with the smaller C, including LOWO actor primary and the five-wave expectation subset | Inner Tuning and Determinism; Erratum 5.1 note | Charter-only modeling lock; `analysis/data_audit.py` remains a data audit and does not fit models |
| The expectation subset always uses the same 1,800 rows, including all ten one-sided wave 5 rows; the without-one-sided rerun applies only to actor primary and participant metrics | Preprocessing and Leakage Controls; Acceptance Criteria; Erratum 5.1 note | Expectation masks at `analysis/data_audit.py:708-714`; observed-row counts at lines 939-945; exact assertions at lines 1109-1122; one-sided construction at lines 560-620 and assertions at lines 1147-1152 |
| Small-wave sensitivity is the unweighted mean of the remaining 18 stored per-wave ROC AUC minus 0.5 scores, with no refitting | Locked Evaluation Specification; Erratum 5.1 note | Small-wave counts constructed at `analysis/data_audit.py:699-705` and exported at lines 846-856; exact assertions at lines 1064-1073 |
| Fold hash serialization is UTF-8, has no header, uses sorted `P####,fold` lines, and includes the final newline | Reproducibility Standards; Erratum 5.1 note | Hash construction at `analysis/data_audit.py:469-474`; expected digest at lines 130-132; verification at lines 1180-1187 |
| New-couple clustered resampling reports a percentile 95 percent interval | Secondary Inference; Erratum 5.1 note | Charter-only inference lock; couple membership source is `analysis/data_audit.py:477-505`; no resampling or model code was added |
| An absent context `field` is its own level before rare-level grouping; 63 rows are missing | Preprocessing and Leakage Controls; Erratum 5.1 note | Context allowlist at `analysis/data_audit.py:97-109`; preprocessing remains approval-gated and no context model was fit |
| Derived indicators occur only in the two sensitivity allowlists | Preprocessing and Leakage Controls | Derived features at `analysis/data_audit.py:91-95`; allowlists at lines 111-120; new exclusivity assertions at lines 1223-1240 |

## Version 5 foundation

| Fourth-audit requirement | Charter section | Data contract evidence |
|---|---|---|
| Couple outer fold k uses both-outside for training, both-in-k for testing, and exactly-one-in-k for dropping | Validation Design; Inner Tuning and Determinism; Charter Decisions; Risk Register | Membership algorithm at `analysis/data_audit.py:477-505`; fold columns and table construction at lines 748-756; exact assertions at lines 1205-1208 |
| Couple inner fold j uses both people outside k and j for training and both in j for validation | Validation Design; Inner Tuning and Determinism | Charter-only implementation lock; no model code was added |
| `fold_a`, `fold_b`, and the clarified `couple_fold` are exported | Validation Design | Construction at `analysis/data_audit.py:748-756`; definition exported at line 973; pair artifact written at lines 1273-1276 |
| Complete-core outer counts are fixed for all five folds | Validation Design | Expected values at `analysis/data_audit.py:122-128`; membership assertions at lines 1205-1208; exported table path at line 1248 |
| All-pair coverage remains 105 of 105; complete-core coverage is 103 of 105 | Validation Design; Acceptance Criteria | Complete-core cell construction at `analysis/data_audit.py:769-789`; assertions at lines 1191-1204 |
| Complete-core empty cells are wave 6 fold 0 and wave 16 fold 3; pure tests total 825 couples and 132 matches | Validation Design; Acceptance Criteria | Assertions at `analysis/data_audit.py:1196-1204` |
| Participant Brier improvement has an exact equal-participant formula | Locked Evaluation Specification; Acceptance Criteria | Charter-only implementation lock; no model was fit |
| Participant deviance is an unweighted participant sum and its row-log-loss equality is only an identity check | Locked Evaluation Specification; Acceptance Criteria | Charter-only implementation lock |
| Participant primary covers 542 complete-core participants and sensitivity covers all 551 | Preprocessing and Leakage Controls; Locked Evaluation Specification; Acceptance Criteria | Counts exported at `analysis/data_audit.py:863-866` and asserted at lines 1075-1076 |
| Spearman excludes waves when predictions or observed rates are constant | Locked Evaluation Specification; Acceptance Criteria | Charter-only implementation lock |
| Expectation sign-flip test is one-sided with the exact numerator, denominator, and minimum p-value | Secondary Inference; Acceptance Criteria | Charter-only implementation lock |
| New-actor and couple analyses use only the L2 logistic pipeline and the four-value C grid | Models and Metrics; Secondary Inference; Inner Tuning and Determinism | Charter-only implementation lock |
| Couple tuning uses pooled inner-validation ROC AUC | Inner Tuning and Determinism | Charter-only implementation lock |
| Trees and forests are restricted to the Milestone 4 LOWO comparison | Models and Metrics; Inner Tuning and Determinism; Milestones | Charter-only implementation lock |
| Actor and couple imputation sensitivities have explicit allowlists | Preprocessing and Leakage Controls; Acceptance Criteria | Derived features at `analysis/data_audit.py:91-95`; allowlists at lines 111-120; policy and exclusivity assertions at lines 1210-1240 |
| Actor medians use training-fold rows; couple medians use training couples separately by gender | Preprocessing and Leakage Controls; Acceptance Criteria | Charter-only implementation lock |
| Calibration uses precomputed grouped wave splits inside every inner-training set | Calibration and Thresholds | Charter-only implementation lock, consistent with the pinned scikit-learn API |
| Logistic threshold comes from the selected C's uncalibrated inner out-of-fold predictions | Calibration and Thresholds | Charter-only implementation lock |
| Couple permutation p-value is `(1 + count(null pooled AUC >= observed)) / 2001` | Secondary Inference | Charter-only implementation lock |
| Scope no longer promises a participant count model | Scope | Charter text |
| Wave 12 exclusion is limited to actor primary, actor AP, and participant primary and secondary metrics | Validation Design; Risk Register; Milestones | Charter text |
| Context benchmark is approval-gated and fully locked if approved | Feature Sets; Preprocessing and Leakage Controls | Context allowlist at `analysis/data_audit.py:97-109` and within `FEATURE_ALLOWLISTS` at lines 111-120 |
| Actor folds remain unchanged and are pinned by counts and SHA-256 | Validation Design; Reproducibility Standards; Charter Decisions | Hash construction at `analysis/data_audit.py:469-474`; expected hash at lines 130-132; assertions at lines 1180-1187 |
| Cross-platform float and byte-identity policy is documented | Reproducibility Standards | Policy exported at `analysis/data_audit.py:1037-1047` |

Line numbers above refer to the delivered V5.2 repository state. Version 5.2 does not change `analysis/data_audit.py`; if a formatter changes line positions, search for the named symbol.

## Required Version 5.2 checks

1. Confirm the status is `Version 5.2 (erratum)` and the Erratum 5.2 note identifies the narrow scope correction and approval-gated context clarifications.
2. Confirm the canonical one-sided-row scope rule appears in all six mapped passages and that no broader `actor and participant analyses/results are rerun` wording remains.
3. Confirm the ten rows remain observed nonmatches for actor and participant analyses, remain excluded from couple models, and remain included in the identical 1,800-row expectation subset.
4. Confirm the context benchmark remains approval-gated and uses the same 8,273 complete-core rows as the actor primary if approved.
5. Confirm the literal string `?` is the explicit missing level for gender, race, race_o, and field before categorical processing; independently verify field / race / race_o contain 63 / 63 / 73 such values and all 63 missing-field rows are core-missing.
6. Confirm `analysis/data_audit.py` is unchanged from Version 5.1, the default audit exits 0, and no raw-data or V4/V5 artifact hash, size, or modification time changes.

## Version 5.1 foundation checks

1. Confirm the smaller-C exact-tie rule covers every logistic tuning step: LOWO actor primary, five-wave expectation subset, actor-grouped, and couple.
2. Confirm the small-wave sensitivity uses the 18 stored per-wave scores without refitting, while the separate wave 12 exclusion remains a rerun without wave 12.
3. Independently reconstruct the fold-hash byte string, including the final newline, and verify `abeb7d4c6606e38611416044cedbdd92cc33cb9c56387ded5b4ab91915aa78dc`.
4. Confirm the new-couple 10,000 clustered resamples produce a percentile 95 percent interval.
5. Confirm the derived-feature assertions restrict all three indicators to `core_imputation_sensitivity` and `couple_imputation_sensitivity`.

## Required re audit checks

1. Record the six V5 artifact modification times, run `.venv/bin/python analysis/data_audit.py`, and verify that the process exits successfully without changing those times.
2. Do not rerun the audit with `--output-dir`. The previously verified command `.venv/bin/python analysis/data_audit.py --output-dir output/audit/v5` is expected to refuse, exit nonzero, and write nothing; the Version 5.2 audit should cite that evidence rather than repeat the output-directory run.
3. Confirm V4 files are unchanged and V5 contains the additional couple membership artifact.
4. Recreate the actor folds from the single-generator algorithm and verify the SHA-256 `abeb7d4c6606e38611416044cedbdd92cc33cb9c56387ded5b4ab91915aa78dc`, the fold sizes 123/114/111/106/97, and 845 pure versus 3,339 mixed pairs.
5. Verify `fold_a` and `fold_b` against participant assignments and confirm that `couple_fold` is the shared test fold or `mixed`.
6. Independently reproduce the complete-core train/test/dropped counts for each outer fold. Confirm that a couple split across two non-k folds is included in outer training.
7. Verify all-pair pure coverage is 105 of 105 and complete-core coverage is 103 of 105, with only wave 6 fold 0 and wave 16 fold 3 empty. Confirm 825 pure complete-core test couples and 132 matches.
8. Confirm six allowlists, three derived features, 55 prohibited categorical `d_` fields, allowed numeric `d_age`, no prohibited or categorical-bin intersection, and that derived features occur only in the two sensitivity allowlists.
9. Check the participant Brier and deviance contracts algebraically, including the distinction between summed row-log-loss improvement and per-wave-averaged actor log loss.
10. Confirm the 542-participant primary, 551-participant sensitivity, and two-sided constant-variable exclusion condition for Spearman waves.
11. Enumerate all 32 expectation sign patterns and confirm the stated one-sided exact p-value and minimum attainable p-value.
12. Confirm the new-actor and couple estimator and tuning restrictions, the pooled couple inner score, and the Milestone 4-only tree and forest scope.
13. Check that the calibration procedure fits each calibrator only inside the applicable inner-training set and uses precomputed `GroupKFold(n_splits=5, shuffle=False)` wave splits.
14. Confirm the logistic F1 threshold uses the selected C's uncalibrated inner out-of-fold scores and the highest-threshold tie rule.
15. Confirm the couple permutation p-value denominator is 2001 and its null statistic is pooled ROC AUC.
16. Check the context benchmark's approval gate, LOWO-only restriction, L2 logistic estimator, categorical handling, rare-field rule, and training-fold numeric imputation.
17. Verify Python 3.14.5, matplotlib 3.11.2, and scikit-learn 1.9.1 are installed and fully represented in `requirements.lock`.
18. Confirm no model has been fitted and no instruction or output modifies `speeddating.arff`, `output/audit/v4`, or `output/audit/v5`.

## Expected V5 data contract highlights

| Check | Expected |
|---|---:|
| Directed rows / participants / waves | 8,378 / 551 / 21 |
| Complete-core / sensitivity participants | 542 / 551 |
| Reciprocal pairs / one-sided rows | 4,184 / 10 |
| Complete-core couple primary | 4,079 pairs |
| Core-missing couple sensitivity | 105 pairs / 29 matches |
| Fold 0 train / test / dropped | 2,475 / 202 / 1,402 |
| Fold 1 train / test / dropped | 2,547 / 180 / 1,352 |
| Fold 2 train / test / dropped | 2,609 / 164 / 1,306 |
| Fold 3 train / test / dropped | 2,692 / 145 / 1,242 |
| Fold 4 train / test / dropped | 2,739 / 134 / 1,206 |
| All-pair pure cells | 105 / 105 |
| Complete-core pure cells | 103 / 105 |
| Empty complete-core cells | wave 6 fold 0; wave 16 fold 3 |
| Pure complete-core tests / matches | 825 / 132 |
| Pure / mixed all pairs | 845 / 3,339 |
| Categorical `d_` columns blocked | 55 |
| Explicit allowlists / derived indicators | 6 / 3 |

## Desired output

1. Executive verdict: READY, READY WITH MINOR REVISIONS, or NOT READY.
2. Reproduced evidence and environment verification.
3. Disposition of every item in the Version 5.2 erratum map, Version 5.1 erratum map, and Version 5 foundation: closed, partial, or open.
4. New findings in descending severity with exact deliverable references.
5. The smallest required changes before the Milestone 4 comparison begins.
6. Remaining grader or team decision gates.
7. A direct statement of which milestone may safely begin.

Do not treat style preferences as defects unless they create analytical ambiguity.
