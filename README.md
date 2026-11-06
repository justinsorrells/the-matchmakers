# The Matchmakers

Machine-learning analysis of self-perception and mutual-match outcomes in the
Columbia speed-dating dataset. The repository contains the frozen Version 5.2
project charter, the audited data contract, Milestone 2 exploratory analysis,
and the Milestone 3 actor logistic baseline.

## Environment

The project uses Python 3.14.5 and the pinned packages in `requirements.lock`.

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.lock
```

## Rebuild Milestones 2 and 3

Existing outputs are protected unless `--force` is supplied explicitly.

```bash
.venv/bin/python -m analysis.run_milestones --milestone all --force
```

Milestone 2 artifacts are written to `output/m2/`; Milestone 3 artifacts are
written to `output/m3/`.

## Verify

```bash
.venv/bin/python analysis/data_audit.py
.venv/bin/python -m unittest discover -s analysis/tests -v
```

The raw `speeddating.arff`, frozen charter, `analysis/data_audit.py`, and files
under `output/audit/` are treated as immutable inputs to Milestones 2 and 3.
