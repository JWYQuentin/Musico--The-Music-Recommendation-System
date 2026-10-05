# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Music recommender on Music4All-Onion

## Goal
A two-stage recommender for a data-science portfolio: a two-tower retrieval model
(PyTorch) that picks a few hundred candidate tracks per user, then a gradient-boosted
ranker (XGBoost, LambdaRank) that orders them. Benchmarked against popularity, item-kNN,
and ALS on a temporal split, with a popularity-bias and genre-fairness analysis and a
small live demo. The owner is learning PyTorch through this project.

## Status
- Phase 1 (setup and data audit): done on the real data, 2026-10-04. The answers to the
  five audit questions are in `reports/decisions.md`.
- Phase 2 (temporal splits, metrics, leakage tests): done, 2026-10-04.
- Phases 3-6: not started. See README.md for the plan.
- `pyproject.toml` lists only what Phases 1-2 use. Add a phase's libraries there when the
  phase starts.

## Data
Music4All-Onion, Zenodo record 6609677, CC BY 4.0. Listening events come from Last.fm.

| File | Contents |
|---|---|
| `data/raw/userid_trackid_timestamp.tsv.bz2` | one row per listen: user, track, time |
| `data/raw/id_ivec256.tsv.bz2` | audio i-vector per track |
| `data/raw/id_lyrics_word2vec.tsv.bz2` | lyrics embedding per track |
| `data/raw/id_genres_tf-idf.tsv.bz2` | genre TF-IDF per track |
| `data/raw/id_tags_dict.tsv.bz2` | Last.fm tags per track |
| `data/interim/events_full.parquet` | all listens, converted |
| `data/processed/events.parquet` | the working subsample: `user_id` (Int64), `track_id` (string), `timestamp` (datetime, stored as milliseconds, values are whole seconds), sorted by user then time |
| `data/processed/splits/{retrieval_train,ranker_train,val,test}.parquet` | the subsample cut by time; same columns |
| `data/processed/events.meta.json`, `data/processed/splits/splits.meta.json` | written beside the data: window and split boundaries, user and listen counts |

- There are no likes, skips, or play durations. A row means "user played track at time".
- Track titles, artists, and genre names are NOT in Onion. They come from the base
  Music4All dataset, requested by email on 2026-10-04. Until it arrives, work on track IDs.
- The events file has a header row and `YYYY-MM-DD HH:MM:SS` timestamps (checked against
  the real file on 2026-10-04). `prepare.sniff` detects both at runtime; if `convert`
  fails, look at the first lines of the file before changing code.

## Evaluation
- A track is relevant for a user in `val` or `test` if they play it in that period and
  did not play it earlier in the window. Repeat plays are not scored.
- Models train on `retrieval_train` and `ranker_train`, are tuned on `val`, and are scored
  on `test` without refitting.
- A model hands `evaluate.score(recs, "val")` a Polars frame with `user_id`, `track_id`,
  `score` (higher is better), with more than K tracks per user. The evaluator drops
  already-played tracks, cuts at K, and returns Recall@K, NDCG@K, coverage@K.
- `score` returns a flat dict: `users`, then `recall@K`, `ndcg@K`, `coverage@K` and
  `short@K` for each K in `configs/eval.yaml`. `short@K` is the share of users left with
  fewer than K tracks after the filter; if it is not near 0, the model handed in too few.
- For `test`, "already played" includes `val` plays, which no model has seen. Averages
  run over users with at least one relevant track; a user with no recommendations scores
  0. Coverage is measured against the tracks in the two train splits.

## Commands
```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m m4a_rec.download            # ~3 GB
python -m m4a_rec.prepare convert     # slow, run once
python -m m4a_rec.prepare subsample
python -m m4a_rec.audit               # writes reports/audit.md
python -m m4a_rec.split               # writes data/processed/splits/
python -m m4a_rec.evaluate            # prints the val and test ground-truth summary
pytest -q
pytest -q tests/test_evaluate.py::test_metrics_hand_checked   # one test
```

- Run everything through the project venv. Without it activated, `python` and `pytest`
  can resolve to another environment and fail with `No module named 'm4a_rec'`; use
  `.venv/bin/python` and `.venv/bin/pytest` instead.
- Run from inside the repo: `config.repo_root` walks up from the working directory to
  find `configs/data.yaml` (or reads `$M4A_ROOT`).
- The `Makefile` wraps the same commands (`make setup`, `make phase1`, `make split`,
  `make test`). No linter or formatter is configured.

## Architecture
- The pipeline is a chain of files, one module per step:
  `download` -> `prepare convert` -> `prepare subsample` -> `audit`, then `split` ->
  `evaluate`. Each step reads the previous step's output from `data/`.
- Every module is pure functions plus a thin `main()`. The functions take frames, paths
  and plain arguments; only `main()` calls `load_config()` and touches `data/`. Tests call
  the functions on synthetic data built in `tests/conftest.py` and never read `data/`.
  New modules follow the same shape so they can be tested the same way.
- `load_config()` merges `configs/data.yaml` and `configs/eval.yaml` into one dict, so
  top-level keys must not collide across config files. It turns `paths` into absolute
  `Path`s and creates those directories. A new config file has to be added to it.
- Time ranges are half-open `(start, end]` everywhere: an event exactly on a boundary
  belongs to the earlier side, in both `prepare.subsample` and `split.split`.

## Rules
- Only `src/m4a_rec/split.py` writes `val` and `test`, and only `src/m4a_rec/evaluate.py`
  reads them. Model code reads the two train splits and gets its numbers from
  `evaluate.score`.
  - `tests/test_leakage.py` enforces this by text search: any module in `src/m4a_rec/`
    outside its allowlist fails if it contains `events.parquet`, `events_full`,
    `val.parquet` or `test.parquet`, even in a comment. So model code may not read the
    full subsample either.
  - The search covers top-level `*.py` files only. If model code goes in a subpackage,
    extend the test to search it.
- Every split is temporal. No random row splits, no features computed from events after
  a split boundary.
- Every tunable number lives in a file under `configs/`, not in code.
- Every design choice gets one entry in `reports/decisions.md`: what, why, alternatives.
- New logic gets a test in `tests/` with hand-checkable numbers. Run `pytest -q` before
  calling anything done.
- `reports/audit.md` is generated. Change `audit.py` and rerun; do not edit the report.
- Use Polars, not pandas. Keep raw data out of git.
- The owner writes the two-tower forward pass and loss by hand in Phase 4. Review that
  code and explain problems; do not rewrite it unasked.
- Start each phase in plan mode and wait for approval before writing code.

# Behavioral guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
