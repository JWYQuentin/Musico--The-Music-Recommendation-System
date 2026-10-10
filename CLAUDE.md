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
- Phase 3 (popularity, item-kNN and ALS baselines): done, 2026-10-05. Results are in
  `reports/baselines.md`; ALS is the one to beat (test Recall@500 = 0.330, NDCG@10 = 0.0265).
- Phase 4 (two-tower retrieval, cold-start test): done, 2026-10-07. Results are in
  `reports/retrieval.md` and `reports/coldstart.md`. `tt_id` and `tt_hybrid` both beat
  ALS (test Recall@100 = 0.139 against 0.130) and are level with each other; `tt_content`
  handles tracks with no listens. Their vectors are saved under `data/processed/models/`
  for Phase 5.
- Phase 5 (XGBoost ranker on `tt_id` candidates): done, 2026-10-10. Results are in
  `reports/ranker.md`. Reordering `tt_id`'s top 500 lifts test NDCG@10 from 0.0325 to
  0.0348 (95% interval on the gain +0.0010 to +0.0035) and Recall@10 from 0.0278 to
  0.0310. In this comparison every model is refitted on both train splits, so these
  numbers are not the Phase 3 and 4 ones. The trained ranker is saved under
  `data/processed/models/ranker/` for Phase 6.
- Phase 6: not started. See README.md for the plan.
- `pyproject.toml` lists only what Phases 1-5 use. Add a phase's libraries there when the
  phase starts.

## Data
Music4All-Onion, Zenodo record 6609677, CC BY 4.0. Listening events come from Last.fm.

| File | Contents |
|---|---|
| `data/raw/userid_trackid_timestamp.tsv.bz2` | one row per listen: user, track, time |
| `data/raw/id_ivec256.tsv.bz2` | audio i-vector per track: 100 numbers, despite the name |
| `data/raw/id_lyrics_word2vec.tsv.bz2` | lyrics embedding per track |
| `data/raw/id_genres_tf-idf.tsv.bz2` | genre TF-IDF per track |
| `data/raw/id_tags_dict.tsv.bz2` | Last.fm tags per track |
| `data/interim/events_full.parquet` | all listens, converted |
| `data/processed/events.parquet` | the working subsample: `user_id` (Int64), `track_id` (string), `timestamp` (datetime, stored as milliseconds, values are whole seconds), sorted by user then time |
| `data/processed/splits/{retrieval_train,ranker_train,val,test}.parquet` | the subsample cut by time; same columns |
| `data/processed/events.meta.json`, `data/processed/splits/splits.meta.json` | written beside the data: window and split boundaries, user and listen counts |
| `data/processed/features.npz` | standardised audio, lyrics and genre features for every training track, with `track_ids` |
| `data/processed/models/<model>/` | a trained two-tower model's `user_vecs.npy`, `track_vecs.npy`, the IDs their rows belong to, and `tracks.faiss`. `tt_id` is fitted on `retrieval_train`, `tt_id_refit` on both train splits |
| `data/processed/recs/<model>.parquet` | the top 1,000 tracks per user from popularity, item-kNN, ALS and `tt_id`, each fitted on both train splits |
| `data/processed/ranker/{train,infer}.parquet` | the ranker's two snapshots: one row per (user, candidate) with every feature; `train` also has `label` |
| `data/processed/models/ranker/` | the trained ranker (`model.json`) and its feature list and tree count (`meta.json`) |

- There are no likes, skips, or play durations. A row means "user played track at time".
- Track titles, artists, and genre names are NOT in Onion. They come from the base
  Music4All dataset, which arrived on 2026-10-10 as `data/raw/music4all.zip` (48 GB). It
  has not been unpacked or checked yet; nothing up to Phase 5 uses it.
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
- `score(recs, split, only_tracks=[...])` scores as if those were the only tracks in the
  catalogue. The cold-start test uses it.
- `evaluate.difference(recs_a, recs_b, split)` gives the gap between two models with a 95%
  interval, from resampling users (`bootstrap` in `configs/eval.yaml`).
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
python -m m4a_rec.baselines tune      # every grid setting on val, ~15 min; writes reports/baselines_tuning.md
python -m m4a_rec.baselines report    # chosen settings on val and test, ~2 min; writes reports/baselines.md
python -m m4a_rec.features            # ~30 s; writes data/processed/features.npz
python -m m4a_rec.retrieval tune tt_id   # or tt_hybrid; ~10 min per run, resumes from reports/twotower_tuning.json
python -m m4a_rec.retrieval report    # tt_id and tt_hybrid on val and test, ~30 min; writes reports/retrieval.md
python -m m4a_rec.coldstart val       # cold-start test on val only, prints, ~10 min
python -m m4a_rec.coldstart report    # the same on val and test; writes reports/coldstart.md
python -m m4a_rec.index data/processed/models/tt_id   # builds and checks the FAISS index
python -m m4a_rec.baselines refit    # the three baselines on both train splits, ~3 min; writes data/processed/recs/
python -m m4a_rec.retrieval refit    # tt_id on both train splits, ~25 min; writes models/tt_id_refit/ and recs/tt_id.parquet
python -m m4a_rec.candidates         # both ranker snapshots, ~7 min; prints each feature's mean in the two
python -m m4a_rec.ranker tune        # the tuning stages on val; resumes from reports/ranker_tuning.json
python -m m4a_rec.ranker ablate      # the chosen setting without each feature group, on val
python -m m4a_rec.ranker val         # every model on val; prints, writes nothing
python -m m4a_rec.ranker report      # val and test, intervals, SHAP; writes reports/ranker.md
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
- Model code gets its data through `interactions.py`: `load_train` reads the train splits
  (and refuses any other name), `build` turns listens into a sparse user-by-track matrix
  with sorted user and track IDs as the row and column order, and `to_frame` turns a
  model's `(users, N)` index and score arrays into the frame `evaluate.score` takes,
  dropping padding and tracks the user played in training. `baselines.py` shows the
  pattern; later models should reuse it so every model shares one index mapping.
- The two-tower code is split by job. `twotower.py` holds the towers and the loss (the
  owner's code). `retrieval.py` trains them (`fit`), stops on a metric handed in as a
  function, and ranks tracks for every user (`top_unseen`, `recommend`). `features.py`
  supplies the content matrix in the same track order as `interactions.build`.
  `coldstart.py` reuses `fit` with some tracks left out. `reporting.py` has the table
  and JSON helpers every report uses.
- The ranker works from two snapshots, defined in `configs/ranker.yaml`. `train`: history
  is `retrieval_train`, candidates and the ALS score come from models fitted on it alone,
  and the label is whether the user first plays the candidate in `ranker_train`. `infer`:
  history is both train splits, the models are the refitted ones, and there is no label.
  `rank_features.py` turns a history and a candidate list into features and knows nothing
  about files or splits; `candidates.py` builds both snapshots with it; `ranker.py` trains
  on `train`, reorders `infer` and hands the lists to `evaluate.score`. A new feature goes
  in `rank_features.GROUPS` and must be computed from the history argument only.
- Every module is pure functions plus a thin `main()`. The functions take frames, paths
  and plain arguments; only `main()` calls `load_config()` and touches `data/`. Tests call
  the functions on synthetic data built in `tests/conftest.py` and never read `data/`.
  New modules follow the same shape so they can be tested the same way.
- `load_config()` merges every file under `configs/` (`data.yaml`, `eval.yaml`,
  `baselines.yaml`, `twotower.yaml`, `ranker.yaml`) into one dict, so
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
- Every file in `reports/` except `decisions.md` is generated. Change the module that
  writes it and rerun; do not edit the reports.
- FAISS and PyTorch abort the process if both are loaded (two OpenMP runtimes on macOS).
  `src/m4a_rec/index.py` is the only module that imports `faiss`; it must not import
  `torch` or any module that does, and its tests run it in a subprocess.
- XGBoost crashes the process if it is loaded after PyTorch, for the same reason.
  `src/m4a_rec/ranker.py` is the only module that imports `xgboost`; it must not import
  `torch` or any module that does (`retrieval`, `twotower`, `coldstart`, `candidates`), so
  it reads other models' output from files. Its tests are in `tests/isolated/`, which
  `pytest` skips and `tests/test_ranker.py` runs in a process of its own.
- XGBoost needs the OpenMP library `libomp`. Here it finds Anaconda's copy, because the
  venv is built on Anaconda's Python; elsewhere it needs `brew install libomp`.
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
