# Two-stage music recommender on Music4All-Onion

A retrieval-then-ranking recommender: a two-tower neural network picks candidate tracks,
and a gradient-boosted model ranks them. Work in progress; Phases 1 to 4 of 6 are done.

## Phase 1: get the data and audit it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                             # 81 tests, a few seconds

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

## Phase 3: baselines

```bash
python -m m4a_rec.baselines tune      # every grid setting on val, about 15 minutes
python -m m4a_rec.baselines report    # the chosen settings on val and test, about 2 minutes
```

Three standard recommenders, fitted on `retrieval_train` and tuned on `val`: the most
popular tracks, item-kNN, and ALS matrix factorisation (the last two from the `implicit`
library). Results on `test`, 10,280 users:

| Model | Recall@10 | NDCG@10 | Recall@100 | Recall@500 | Coverage@10 |
|---|---|---|---|---|---|
| Popularity | 0.0041 | 0.0060 | 0.0291 | 0.0936 | 0.1% |
| Item-kNN | 0.0203 | 0.0262 | 0.1044 | 0.2754 | 20.7% |
| ALS | 0.0234 | 0.0265 | 0.1300 | 0.3298 | 22.0% |

For scale, random recommendations score Recall@10 = 0.00015 on `val`. The numbers are
low because only tracks new to the user count, out of a catalogue of 56,000. ALS and
item-kNN are level at the top of the list; ALS finds more of a user's new tracks deeper
down, and popularity recommends almost the same few tracks to everyone.

All four K values are in `reports/baselines.md`, the tuning grid in
`reports/baselines_tuning.md`, and the reasons for each setting in `reports/decisions.md`.

## Phase 4: two-tower retrieval and cold start

```bash
python -m m4a_rec.features                 # content features, about 30 seconds
python -m m4a_rec.retrieval tune tt_id     # tuning on val, about 10 minutes a run; also tt_hybrid
python -m m4a_rec.retrieval report         # tt_id and tt_hybrid on val and test, about 25 minutes
python -m m4a_rec.coldstart report         # the cold-start test, about 10 minutes
python -m m4a_rec.index data/processed/models/tt_id   # FAISS index for the demo
```

A two-tower model in PyTorch: one network turns a user into a vector, another turns a
track into a vector, and a track is recommended when the two line up. Three versions
differ in what the track network is given: the track's ID (`tt_id`), its ID plus audio,
lyrics and genre features (`tt_hybrid`), or the features alone (`tt_content`). All are
trained on `retrieval_train` with an in-batch softmax loss that corrects for track
popularity.

Results on `test`, 10,280 users, beside the Phase 3 baselines:

| Model | Recall@10 | NDCG@10 | Recall@100 | Recall@500 | Coverage@10 |
|---|---|---|---|---|---|
| Popularity | 0.0041 | 0.0060 | 0.0291 | 0.0936 | 0.1% |
| Item-kNN | 0.0203 | 0.0262 | 0.1044 | 0.2754 | 20.7% |
| ALS | 0.0234 | 0.0265 | 0.1300 | 0.3298 | 22.0% |
| `tt_id` | 0.0258 | 0.0304 | 0.1390 | 0.3438 | 28.9% |
| `tt_hybrid` | 0.0261 | 0.0307 | 0.1390 | 0.3450 | 28.8% |

Both two-tower models beat ALS: by 7% on Recall@100 and about 15% on NDCG@10. The
popularity correction in the loss mattered more than any other setting; without it the
model scored below item-kNN on `val`. Content features add nothing for tracks that
already have listens.

They matter for tracks that have none. In the cold-start test, 2,808 tracks (5%) are
removed from training entirely and models are scored on those tracks alone. Results on
`test`, 4,004 users:

| Model | Recall@10 | NDCG@10 | Recall@100 |
|---|---|---|---|
| `tt_content` | 0.0575 | 0.0313 | 0.2804 |
| Content similarity, no training | 0.0369 | 0.0197 | 0.1983 |
| Random | 0.0021 | 0.0011 | 0.0369 |
| Popularity, item-kNN, ALS | 0 | 0 | 0 |

The content-only model finds 28% of a user's new held-out tracks in its top 100, and
training it adds about 41% over comparing raw features directly.

Full tables are in `reports/retrieval.md` and `reports/coldstart.md`, the tuning runs in
`reports/twotower_tuning.md`.

## Layout

```
configs/data.yaml      every Phase 1 setting
configs/eval.yaml      split lengths and K values
configs/baselines.yaml baseline grids and chosen settings
configs/twotower.yaml  two-tower settings, tuning stages, cold-start hold-out
src/m4a_rec/           download.py, prepare.py, audit.py, split.py, evaluate.py,
                       interactions.py, baselines.py, features.py, twotower.py,
                       retrieval.py, coldstart.py, index.py, reporting.py
tests/                 run on synthetic data in the real file layout
reports/               decisions.md, and generated reports for the audit, baselines,
                       retrieval and cold start
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
