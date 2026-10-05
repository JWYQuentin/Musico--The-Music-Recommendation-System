# Two-stage music recommender on Music4All-Onion

A retrieval-then-ranking recommender: a two-tower neural network picks candidate tracks,
and a gradient-boosted model ranks them. Work in progress; Phases 1 and 2 of 6 are done.

## Phase 1: get the data and audit it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                             # 27 tests, a few seconds

python -m m4a_rec.download            # about 3 GB from Zenodo
python -m m4a_rec.prepare convert     # slow: decompresses 2.2 GB of bz2
python -m m4a_rec.prepare subsample   # seconds
python -m m4a_rec.audit               # writes reports/audit.md and 3 figures
```

Disk needed: about 3 GB for downloads plus a few GB for the converted Parquet file.

Then read `reports/audit.md` and answer these before Phase 2:

1. **Are the last months complete?** If the monthly chart tails off, set `window_end`
   in `configs/data.yaml` to the last full month and rerun `subsample` and `audit`.
2. **How many users did the 50-listen minimum drop?** Adjust `min_user_listens` if it
   removes most of them.
3. **How large is the repeat-play share?** A high share means evaluation should score
   only tracks that are new to each user.
4. **How skewed is popularity?** This sets how hard the popularity baseline is to beat.
5. **What share of tracks has audio, lyrics, and genre features?** This bounds the
   cold-start experiment.

Record the answers in `reports/decisions.md`.

## Phase 2: splits and evaluation

```bash
python -m m4a_rec.split               # writes data/processed/splits/, seconds
python -m m4a_rec.evaluate            # prints what val and test will be scored against
```

The 13-month window is cut by time into four parts:

| Split | Period | Listens | Used for |
|---|---|---|---|
| `retrieval_train` | 2019-02-20 to 2020-01-24 | 15,105,295 | training the retrieval model and baselines |
| `ranker_train` | 28 days to 2020-02-21 | 1,204,592 | training the ranker |
| `val` | 14 days to 2020-03-06 | 586,655 | tuning |
| `test` | 14 days to 2020-03-20 | 584,036 | the reported numbers |

A model is scored only on tracks that are new to the user: played in the period, not
played earlier in the window. That is about 128,000 (user, track) pairs per period, a
median of 8 per user. Metrics are Recall@K, NDCG@K and catalogue coverage at
K = 10, 20, 100 and 500.

Every model hands `evaluate.score(recs, "val")` a Polars frame with `user_id`,
`track_id` and `score`. Only `evaluate.py` reads `val` and `test`; `tests/test_leakage.py`
fails if any other module names those files.

## Layout

```
configs/data.yaml      every Phase 1 setting
configs/eval.yaml      split lengths and K values
src/m4a_rec/           download.py, prepare.py, audit.py, split.py, evaluate.py
tests/                 run on synthetic data in the real file layout
reports/               audit.md (generated), decisions.md
CLAUDE.md              project brief and rules for Claude Code
```

## Plan

| Phase | Work |
|---|---|
| 1 | Setup and data audit |
| 2 | Temporal splits, Recall@K / NDCG@K / coverage, leakage tests |
| 3 | Baselines: popularity, item-kNN, ALS |
| 4 | Two-tower retrieval in PyTorch, FAISS index, cold-start test |
| 5 | XGBoost LambdaRank ranker, SHAP |
| 6 | Popularity-bias and genre-fairness analysis, demo, write-up |

## Data and citation

Music4All-Onion (Moscati et al., CIKM 2022), Zenodo record 6609677, CC BY 4.0.
Track metadata from Music4All (Santana et al., IWSSIP 2020), available on request.
