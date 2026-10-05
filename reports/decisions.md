# Decisions log

One entry per design choice: what was chosen, why, and what was rejected.

## Phase 1

**Dataset: Music4All-Onion, not Yambda.**
Why: the demo must show real song titles and genres, and Yambda's track IDs are anonymous.
Cost: no organic/recommendation flag, so the feedback-loop analysis is replaced by a
popularity-bias and genre-fairness analysis.

**Window: the last 13 months of listens.**
Why: Phase 2 needs 10 months for retrieval training, 1 month for ranker training, and
2 weeks each for validation and test; 13 leaves a little slack.
To check in the audit: whether the final months are complete. If the last bars of the
monthly chart drop off, set `window_end` in `configs/data.yaml` to the last full month.

**Users: at least 50 listens in the window; all 14,366 who qualify.**
Why: a user tower cannot learn from a handful of plays. Only 17,180 of the dataset's
119,140 users are active in the final 13 months and 14,366 of them reach 50 listens, so
the `n_users: 20000` cap never binds and no sampling takes place. The cap and seed stay
in `configs/data.yaml` so a different window still gives a reproducible, laptop-sized
sample. Whole users are kept (not rows), so each user's history stays intact.
Rejected: sampling rows, which breaks sequences; taking the most active users, which
biases toward heavy listeners.
Corrected 2026-10-04: this entry used to say "then a seeded random 20,000", written
before the real data showed how few users are active in the window.

**No item filter yet.**
Why: rare tracks are the cold-start test in Phase 4. Filtering them now would remove it.
Audit result: the window has almost no rare or new tracks, so this filter would remove
little either way and the cold-start test needs a different source. See answer 5 below.

**Two-step preparation (convert, then subsample).**
Why: bz2 decompression is slow and single-threaded. Converting once to Parquet makes
every later change to the subsample settings take seconds.

**Audit is a script, not a notebook.**
Why: it reruns identically whenever the subsample changes, and its numbers are unit-tested.

**Matrix density counts unique (user, track) pairs.**
Why: density is the share of filled cells in the user-by-track matrix. Dividing listens
by users x tracks counted every repeat play as another cell and reported 2.17%; with
unique pairs it is 0.66%.
Rejected: listens / (users x tracks), the original formula in `audit.py`.

## Phase 1 audit answers (2026-10-04)

From `reports/audit.md` on the real data: 17,480,578 listens, 14,366 users, 56,203
tracks, 2019-02-20 to 2020-03-20. Numbers marked "one-off check" were computed by hand
on the same files that day and are not in the generated report.

**1. The last months are complete; `window_end` stays `null`.**
Why: the March 2020 bar is low only because the data stops at 2020-03-20 12:59. Daily
volume is 36-46k listens through 2020-03-19, the same as February (one-off check), and
month-to-month variation over complete months is 0.08.
Rejected: cutting the window at 2020-02-29, which would discard 19 good days.

**2. `min_user_listens` stays at 50.**
Why: it drops 2,814 of 17,180 active users (16%), and those users account for 0.26% of
listens in the window (one-off check). Among kept users the 5th percentile is 83 listens
and the median is 747.
Rejected: lowering it to 20 or 10, which adds about 1,000 to 1,500 users with too few
plays to split into train, validation, and test.

**3. Repeat plays are 69.4% of listens, so evaluation scores only tracks new to each user.**
Why: a model that replays a user's history would look strong without recommending
anything. In a trial cut of the final two weeks, 34% of (user, track) pairs were new to
the user: about 128,000 pairs, with 11,689 users active in those two weeks (one-off
check). That is enough to score.
Rejected: scoring all test listens, which mostly measures memorising history.
Settled in Phase 2: no all-listens number is reported; see "Relevant means new within
the window" below.

**4. Popularity skew is mild; the popularity baseline should be beatable.**
Why: the top 1% of tracks take 13.2% of listens, the Gini coefficient is 0.618, and only
1.5% of tracks (846) have fewer than 5 listens. The median track has 150 listens
(one-off check). The 56,203-track catalogue is curated, so it lacks the long tail of a
full streaming catalogue.

**5. Feature coverage is near-complete, but there is no natural cold-start set.**
Why: audio, lyrics, and genre files each have a row for 100% of tracks; tags cover 99.4%
of tracks and 99.8% of listens. A row is not always usable: 9.9% of tracks (5,550) have
an all-zero lyrics vector, against 0 for audio and 14 for genre (one-off check). No track
is first heard in the final six weeks, and only 558 first appear anywhere in the window
(one-off check).
To settle in Phase 4: the cold-start test must hold tracks out of training artificially,
and the lyrics tower input needs a rule for the all-zero vectors.

## Phase 2: splits and evaluation (2026-10-04)

**Four temporal splits, counted back from the last timestamp in whole weeks.**
`test` is the final 14 days, `val` the 14 days before it, `ranker_train` the 28 days
before that, and `retrieval_train` everything earlier (about 11 months). Boundaries are
2020-01-24, 2020-02-21 and 2020-03-06, all at 12:59:51 on a Friday.
Why: whole weeks give every period the same weekday mix. The ranker needs its own period
so its training labels are listens the retrieval model never saw.
An event exactly on a boundary goes to the earlier split, as in `prepare.subsample`.
Rejected: calendar-month boundaries, which give periods of unequal length and weekday mix.

**Relevant means new within the window.**
A track counts for a user in a period if they play it then and did not play it earlier
in the 13-month window. This gives 127,889 relevant pairs over 10,423 users in `val` and
128,275 over 10,280 in `test`, median 8 per user.
Why: repeat plays are 69.4% of listens, and scoring them rewards memorising history.
Caveat: 45% of these pairs are tracks the user had played before the window opened
(one-off check against the full history), so the task is partly rediscovery. Models
never see that older history and cannot exploit it.
Rejected: "never played in the whole Last.fm history", which is stricter but leaves about
70,000 pairs per period and makes the evaluator filter on history no model can see.

**Models are fitted once and never refitted on validation data.**
They train on the two train splits, are tuned on `val`, and are scored on `test` as they
are. Only `split.py` writes `val` and `test`, and only `evaluate.py` reads them.
Why: one rule that is easy to test, and no model code ever touches held-out listens.
Cost: at test time every model is two weeks stale.
Rejected: refitting on train plus `val` before the test run, which is common but means
model code reads validation data.

**The evaluator drops already-played tracks before cutting the list at K.**
Why: every model is filtered identically, and a model cannot see validation-period plays
at test time. Models hand in more than K tracks per user; the `short@K` figure reports
the share of users left with fewer than K.
Rejected: trusting each model to filter its own list, which lets a filtering bug look
like a modelling difference.

**Metrics: Recall@K, binary NDCG@K, coverage@K at K = 10, 20, 100, 500.**
Recall divides hits by all of the user's relevant tracks. NDCG's ideal list has
min(K, relevant) hits at the top. Both are averaged over users with at least one
relevant track; a user with no recommendations scores 0. Coverage is the share of the
56,189 training tracks that appear in at least one evaluated user's top K.
Why these K: 10 and 20 describe the list a user sees; 100 and 500 describe the candidate
set the retrieval stage hands to the ranker.
Reference points on `val` (one-off checks): uniformly random recommendations score
Recall@10 = 0.00015; a perfect list scores NDCG@10 = 1 but Recall@10 = 0.82, because
users with more than 10 relevant tracks cannot be fully covered in 10 places.
Rejected: Recall divided by min(K, relevant), which hides that ceiling; graded relevance
by play count, which brings repeat plays back in.

**Cold-start is a separate Phase 4 experiment.**
The main splits keep every track. In Phase 4 the two-tower model is trained a second
time with a random 5% of tracks removed and scored on those tracks.
Why: no track is naturally new in the final weeks, and holding tracks out of the main
splits would put a hole in every headline number.
Rejected: removing 5% of tracks from all training data now.

## Phase 3: baselines (2026-10-05)

Numbers are from `reports/baselines_tuning.md` (`val`) unless they say `test`.

**Baselines are fitted on `retrieval_train` only.**
Why: it is the data the two-tower model gets in Phase 4, so retrieval comparisons are
like-for-like, and these fitted models can supply candidates for ranker training in
Phase 5 without having seen its labels.
Cost: at validation time the baselines are four weeks staler than they could be. They can
recommend the 56,166 tracks in `retrieval_train`; coverage is still measured against the
56,189 tracks in both train splits.
To settle in Phase 5: whether the finished two-stage system also needs baselines fitted
on both train splits as a like-for-like opponent (`train_splits` in
`configs/baselines.yaml`).
Rejected: fitting on both train splits now.

**Item-kNN and ALS come from the `implicit` library.**
Why: one library and one calling pattern for both models, and the least code to maintain.
Rejected: a hand-written item-kNN in SciPy, where every step can be checked by hand but
which is slower and about 40 lines longer.

**Selection rule: the setting with the highest NDCG@10 on `val`.**
Why: the top-10 list is what the finished system is judged on.
Caveat: within each model the best two settings are often 0.0002 or less apart, which is
probably noise. ALS is the one exception to the rule; see its entry below.

**Popularity: distinct listeners over the last 180 days of training.**
Why: ranking by listeners beats ranking by total plays at every window length (NDCG@10
0.0061 against 0.0046 at best), because repeat plays let a few heavy listeners lift a
track. The window matters less: 180 days and all of training are tied (0.0061 and
0.0060), and 90 and 28 days are worse (0.0046).
Rejected: total plays; shorter windows.

**Item-kNN: cosine similarity on log(1 + plays), 500 neighbours per track.**
Why: log weighting beats binary (0.0261 against 0.0245 at best). The neighbour count
barely matters: NDCG@10 runs from 0.0241 to 0.0261 across 50, 200 and 500. `implicit`
counts the track itself as one of the neighbours.
Rejected: 200 neighbours, a near-tie (0.0259) with slightly better Recall@500 (0.281
against 0.278); 50 neighbours, which leaves 1.5% of users with fewer than 500 tracks.

**ALS: 256 factors, regularization 0.1, alpha 10, 15 iterations; confidence is alpha x log(1 + plays).**
This is not the NDCG@10 winner. With 256 factors, alpha 1 scores NDCG@10 = 0.0275 and
alpha 10 scores 0.0265, but alpha 10 has Recall@100 = 0.130 against 0.119 and
Recall@500 = 0.330 against 0.304.
Why: Phase 4 compares the two-tower model on Recall@100 and Recall@500, and ALS is the
strongest opponent there. The owner chose the setting that is strongest at that depth and
accepted a 4% lower NDCG@10.
Other findings: alpha 40 is clearly worse; regularization (0.01 or 0.1) makes no visible
difference; more factors help, with each doubling giving about half the gain of the one
before (Recall@500 at alpha 10: 0.304, 0.323, 0.330 for 64, 128, 256).
Not tuned: the log weighting and the 15 iterations.
Rejected: alpha 1, the rule's pick; 512 factors, not tried because the gains were
already halving.

**Each model hands in 1,000 tracks per user, already cleared of the user's training plays.**
Why: the largest K is 500, and the evaluator also drops plays the model has not seen
(`ranker_train` and, for `test`, `val`). For popularity and ALS, short@500 equals the
share of users with no recommendations at all on both `val` and `test`, so 1,000 is
enough. The evaluator still does its own filtering.
Rejected: unfiltered lists, which would have to be far longer for heavy listeners.

**Users who are not in `retrieval_train` get no recommendations.**
59 users first appear after it ends. They are 0.51% of the users scored on `val` and
0.53% on `test`, and they score 0 for every model.
Why: no model has seen them, and a fallback would be an extra rule for half a percent of
users.
Rejected: giving unknown users the popularity list.

**`test` is scored once, after the settings are frozen.**
`tune` reads `val` only. `report` was run on 2026-10-05 after the owner confirmed the
settings, and rerun with the same settings only to confirm that the numbers repeat. They
do, to the 4 places reported; a float mean differs in its 16th digit from run to run, so
`reports/baselines.json` keeps 6 places.
Result on `test`: popularity, item-kNN and ALS score NDCG@10 = 0.0060, 0.0262 and 0.0265
and Recall@500 = 0.094, 0.275 and 0.330, each within 0.003 of its `val` figure.
Rejected: looking at `test` while tuning.
