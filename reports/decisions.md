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
`reports/baselines.json` keeps a fixed number of places: 10 since Phase 4. The original 6
made one cell print differently when the saved numbers were read back.
Result on `test`: popularity, item-kNN and ALS score NDCG@10 = 0.0060, 0.0262 and 0.0265
and Recall@500 = 0.094, 0.275 and 0.330, each within 0.003 of its `val` figure.
Rejected: looking at `test` while tuning.

## Phase 4: two-tower retrieval and cold start (2026-10-07)

Numbers are from `reports/twotower_tuning.md` (`val`) unless they say `test`. Recall@100
for ALS on `val` is 0.130.

**Three two-tower models that differ only in what the track tower is given.**
`tt_id` gets the track ID, `tt_hybrid` the ID plus content features, `tt_content` the
content features alone. The user tower is one trainable vector per user in all three.
Why: each answers one question. `tt_id` has the same information as ALS, so it checks the
training; `tt_hybrid` shows whether content helps for known tracks; `tt_content` can
score a track nobody has played.
Rejected: per-feature ablations; a user tower built from listening history, which would
handle new users, but only 0.5% of scored users are new.

**The owner wrote the towers' forward passes and the loss.**
`UserTower.forward`, `TrackTower.forward` and `in_batch_softmax_loss` in
`src/m4a_rec/twotower.py`, against tests with hand-worked numbers. On the real data their
loss and Recall@100 match a separately written reference to four decimals over the first
four epochs.

**A score is the dot product of two unit-length vectors, divided by a temperature of 0.1.**
Why: unit vectors keep every score between -1 and 1, so popularity cannot enter through
vector length, and the temperature sets how sharply the loss separates right from wrong.
At 128 numbers per vector, 0.1 gives Recall@100 = 0.138, 0.05 gives 0.120 and 0.2 gives
0.115.

**Loss: in-batch softmax with the log-q correction.**
In a batch of 4,096 (user, track) pairs, each user's own track is the right answer and
the other tracks in the batch are the wrong ones. The log of each track's share of the
training pairs is subtracted from its scores.
Why: popular tracks turn up as wrong answers in proportion to their popularity, so the
plain loss pushes them down too far. Without the correction `tt_id` scores Recall@100 =
0.094 and puts 72% of the catalogue in someone's top 10; with it, 0.138 and 30%. This
was the largest effect of any setting.
Rejected: the plain loss.

**Duplicate masking stays on although it changes nothing measurable.**
A wrong answer that is the same track as the right one is ignored. Recall@100 is 0.1381
with and without it: with 56,166 tracks the same track rarely appears twice in a batch.
Why keep it: it is the correct form of the loss and costs nothing.

**Training examples: every unique (user, track) pair once per epoch.**
4.8 million pairs from `retrieval_train`.
Rejected: drawing pairs in proportion to log(1 + plays), which scores 0.136 against
0.142.

**256 numbers per vector, Adam with learning rate 0.003.**
Recall@100 is 0.129, 0.138 and 0.142 for 64, 128 and 256. 256 is the largest size tried
and the size ALS uses, so the comparison is like-for-like. Learning rates 0.003 and 0.01
give 0.138 and 0.137.
Embedding rows start at about length 1 (standard deviation 1/sqrt(size)). With
PyTorch's default, about sqrt(size), each optimizer step is a much smaller turn of the
vector and `tt_id` reached only Recall@100 = 0.002 after two epochs.
Rejected: 512, not tried because the gain had already halved.

**Training stops on Recall@100 on `val`: patience 3, at most 30 epochs.**
After every epoch the model's recommendations go through `evaluate.score`; model code
never reads `val` itself. Runs stopped after 9 to 20 epochs of about 25 to 55 seconds.
Why Recall@100: this model's job is to hand candidates to the ranker.
Caveat: the kept epoch is the best of several noisy `val` readings, so `val` figures are
slightly flattering. `test` is the clean number.

**Settings are tuned in stages, not as a full grid.**
Each stage tries its values on top of the best setting so far: 11 runs for `tt_id`, 4
for `tt_hybrid`.
Why: the full grid is 144 runs of about 9 minutes each.
Cost: settings in different stages are never varied together.
The loss corrections were moved to the first stage after the first run showed the plain
loss far behind ALS.

**Content features: audio (100 numbers), lyrics (300), genres (685) and a has-lyrics flag.**
Audio and lyrics columns are scaled to mean 0 and standard deviation 1; genre rows are
scaled to length 1. The audio vector has 100 numbers despite the "256" in its file name.
Lyrics rule, left open in Phase 1: the 9.9% of tracks with an all-zero lyrics vector are
left out when the scaling is measured, keep zeros afterwards, and get has-lyrics = 0.
The scaling is measured over every training track, including the cold-start hold-out:
these numbers describe the track itself and exist before anyone plays it.
Rejected: Last.fm tags. Listeners add them after release, so a new track has none, and
using them would leak popularity into the cold-start test.

**Content part of the track tower: one hidden layer of 256 with dropout 0.2, added to the ID vector before scaling.**
The four settings tried (hidden 256 or 512, dropout 0 or 0.2) give Recall@100 between
0.136 and 0.141.
Not tried: joining the two parts side by side and adding another layer.

**Finding: content features add nothing for tracks with play history.**
`tt_hybrid` scores Recall@100 = 0.141 against 0.142 for `tt_id` on `val`, and both score
0.139 on `test`.

**Cold-start test: a seeded random 5% of training tracks (2,808) lose all their listens.**
`tt_content` is trained without them, with the hybrid's setting, and stops on
cold-start Recall@100. It is then scored with only the held-out tracks as candidates and
as relevant tracks (`evaluate.score(..., only_tracks=...)`), under the usual rule: played
in the period, not played before. 4,044 users can be scored this way on `val`.
Why artificial: no track is naturally new in the final weeks (Phase 1, answer 5).
Two references need no training: `content_sim` recommends the held-out tracks closest to
the average of what the user played, with the three kinds of feature weighted equally;
`random` orders them with no information. Popularity, item-kNN and ALS cannot recommend a
track with no listens and score 0.
On `test`, with 4,004 users, Recall@100 is 0.280 for `tt_content`, 0.198 for `content_sim`
and 0.037 for `random`, so training adds about 41% over using the features directly. On
`val` the three score 0.301, 0.201 and 0.038.
Rejected: scoring the held-out tracks against their removed training listens, which
gives more pairs but drops the rule that the test period comes after training.

**Finding: without the track ID the model is much weaker on known tracks.**
Scored on all tracks, `tt_content` reaches Recall@100 = 0.092 on `test` against 0.139 for
`tt_id`.

**Recommendations are computed by scoring every track; the FAISS index is exact and separate.**
Why: with 56,166 tracks a full score matrix per 1,000 users is instant, and it lets each
user's played tracks be masked before the top 1,000 are taken. The index is for the
Phase 6 demo and is checked against plain numpy.
FAISS and PyTorch abort the process when loaded together on macOS, so the index is built
by its own command and tested in a subprocess.
Rejected: an approximate index, which only pays off for far larger catalogues.

**Only the user and track vectors are saved, not the model weights.**
Why: the vectors are all that Phase 5 and the demo read.

**Training on the Apple GPU is not exactly repeatable, and that is accepted.**
The same setting gives slightly different numbers from run to run: `tt_hybrid` scored
Recall@100 = 0.1409 on `val` in tuning and 0.1414 in the report run, and `tt_content`
scored 0.291 and 0.301 on the cold-start `val` test. The cold-start figure moves more
because its kept epoch is picked on a noisier metric from 4,044 users.
Why accepted: no difference is large enough to change a conclusion, and the CPU, which
does repeat exactly, takes twice as long per epoch.
Cost: rerunning `report` will not reproduce the tables to the last digit.

**`test` was scored once, after the owner confirmed the settings on 2026-10-07.**
`tune` and `coldstart val` read `val` only. The Phase 3 baselines were rescored in the
same session with their frozen settings, only to save their results with more decimal
places; `reports/baselines.md` came out unchanged.
Result on `test`, 10,280 users:

| Model | Recall@10 | NDCG@10 | Recall@100 | Recall@500 |
|---|---|---|---|---|
| ALS | 0.0234 | 0.0265 | 0.1300 | 0.3298 |
| `tt_id` | 0.0258 | 0.0304 | 0.1390 | 0.3438 |
| `tt_hybrid` | 0.0261 | 0.0307 | 0.1390 | 0.3450 |

Both two-tower models beat ALS on every measure: by 7% on Recall@100, 4 to 5% on
Recall@500 and 14 to 16% on NDCG@10. The margins are smaller than on `val` (9% and 20 to
21%), as expected when the kept epoch is picked on `val`.
For Phase 5: `tt_id` and `tt_hybrid` are equally good candidate generators. `tt_id` is
the simpler of the two.

## Phase 5: ranker (2026-10-08)

Numbers are on `val` unless they say `test`; tuning runs are in `reports/ranker_tuning.md`.

**The ranker is trained in one snapshot and used in a later one.**

| | Snapshot `train` | Snapshot `infer` |
|---|---|---|
| History the models and features see | `retrieval_train` | `retrieval_train` + `ranker_train` |
| Period being predicted | `ranker_train` | `val`, then `test` |
| Candidates from | `tt_id` (Phase 4) | `tt_id` refitted on both splits |
| ALS score from | ALS fitted on `retrieval_train` | ALS fitted on both splits |
| Label | 1 if the user first plays the candidate in `ranker_train` | none; only the evaluator has it |

The same code builds both; only the history it is handed differs.
Why: everything the ranker is given must come from before the period it predicts, in
training as in use. A retrieval model that had already seen `ranker_train` would score
the tracks a user played there highly because it was trained on them, and the ranker
would learn to trust a signal it does not get on `val`.
Rejected: training the ranker on candidates from the refitted model; splitting
`ranker_train` rows at random.

**Every opponent is refitted on both train splits, with its Phase 3 or 4 setting.**
Why: the two-stage system sees `ranker_train`. Against models that stopped four weeks
earlier, fresher data would be counted as the ranker's gain. The fresher data alone
raises NDCG@10 by 3.8% for popularity, 5.3% for item-kNN, 8.6% for ALS and 8.9% for
`tt_id` (0.0320 to 0.0349).
The refitted `tt_id` stops on `val` Recall@100 as in Phase 4; it kept epoch 21 at 0.1510.
Rejected: tuning the opponents again on the larger data, a second round of choices made
on `val`.

**Candidates are `tt_id`'s top 1,000 unplayed tracks per user. ALS is a feature, not a second source.**
In the `train` snapshot the top 100, 200, 500 and 1,000 hold 15%, 22%, 36% and 49% of
the tracks a user goes on to play for the first time. The ranker cannot recommend a track
that is not a candidate, so these are its ceiling.
Why `tt_id`: it and `tt_hybrid` are level, and it is the simpler one.
Rejected: pooling candidates from several models, which would raise the ceiling but mix
the ranker's gain with the wider pool's.

**The label uses the evaluator's rule: 1 if the user plays the candidate in the period and had not played it before.**
124,461 of 14.3 million candidate rows are positive (0.87%). A user with no positive
among the candidates being reordered is left out of training, because a ranking loss
compares a user's candidates with each other: 10,604 of 14,307 users remain at 500
candidates.
Rejected: labels graded by play count; adding the tracks a user played that retrieval
missed, which the ranker never meets in use.

**19 features in four groups, all computed from the snapshot's history.**

| Group | Features |
|---|---|
| models | the candidate's rank and score from `tt_id`, and from ALS |
| track | listeners overall, in the last 28 and the last 7 days; share of its listeners who played it in the last 28 days; share of those who were playing it for the first time |
| user | tracks played; plays in the last 28 days; share of recently played tracks that were new; how popular their tracks are; days since last play |
| pair | similarity to the user's last 28 days of listening in `tt_id` space; audio, lyrics and genre similarity to their history; the track's popularity minus the user's usual |

Scores are standardised within each user's list, because scores from separately trained
models are not on one scale. Counts enter as logs and shares, because the `infer`
snapshot has four more weeks of history. A value that cannot be computed is missing, not 0.
Rejected: raw counts; artist features, which wait on the Music4All metadata.

**`track_age_days` was dropped after the drift check.**
`candidates` prints each feature's mean in the two snapshots. Days since a track was
first heard ran from 319 to 338 for 99% of `train` candidates and from 347 to 366 in
`infer`: nearly every candidate dates from the start of the window, so the feature
measured the window's length, and its two ranges do not overlap. A split learned on one
would mean something else on the other. Every other feature's mean moves by under 5%,
except `user_days_since_last` (15.4 to 17.0 days), which shifts only for users who had
already stopped listening.

**Candidates below the cut-off keep their retrieval order, and so do candidates the ranker scores the same.**
The ranker reorders a user's first `n_candidates`; the rest follow, so every list still
has 1,000 tracks and Recall@500 stays comparable. `rerank` hands the evaluator minus each
track's place in the final list, not the model's raw score.
Why: a model of a few shallow trees gives many candidates exactly the same score, and the
evaluator orders equal scores by track ID, which is arbitrary. In a run with 1,000
candidates and depth-1 trees the fix raised NDCG@10 from 0.0298 to 0.0357 at 25 trees;
from 200 trees on it moved no result by more than 0.0001.

**The number of trees is picked on `val` through the evaluator.**
A model is trained to 600 trees and its lists at every 25 go through `evaluate.score`;
the count with the best NDCG@10 is kept.
Why: it is the rule the two-tower model uses for epochs, and it avoids carving a random
validation set out of the training rows.
The first run, at depth 6, scored every 50 trees and peaked at 50, the first count, so
the step was halved. Deep trees peak early (25 to 50 trees at depth 8) and shallow ones
later (150 to 300 at depth 1). No setting's best count is the last one.
Caveat: as in Phase 4, the kept count is the best of several `val` readings.

**Pairing method `mean`, fixed and not tuned, because `topk` is unstable here.**
LambdaRank learns from pairs of a user's candidates with different labels. `topk`, the
XGBoost default, uses only pairs that involve one of the user's current top 32; `mean`
samples a pair for every candidate.
A full tuning pass with `topk` came first. It picked 200 candidates and depth-1 trees at
NDCG@10 = 0.0394, with a smooth curve. The ablation then retrained that setting without
one feature group at a time:

| Depth 1, 200 candidates | `topk` | `mean` |
|---|---|---|
| Without the pair features | 0.0342 at 25 trees, then 0.023 to 0.027 | 0.0359, rising smoothly |
| Without the model scores | 0.0312 at 25 trees, then jumping between 0.019 and 0.031 | 0.0339, rising smoothly |
| Only `tt_rank` and `tt_z` | 0.0348, then down to 0.0329 by 600 trees | not run |

Retrieval order alone scores 0.0349, so a `topk` ranker given nothing but the retrieval
rank ends up worse than not reordering at all.
Likely reason, not verified: with about 1% of candidates positive, `topk` learns mostly
from the few candidates on top at that moment, and pushing those down changes which ones
are on top.
Why `mean`: on the full feature set the two are level (0.0393 for `topk` against 0.0392
at depth 1 and 500 candidates), and `mean` stays stable when features are removed.
Not tried: `topk` with a cut-off as long as the list, which is one more setting to tune.

**Tuned in stages on `val`, selecting on NDCG@10: 16 runs of one to four minutes.**
Order: tree depth with learning rate, a lower learning rate, candidates reordered,
objective. `tt_id` alone scores NDCG@10 = 0.0349.

| Stage | Result (NDCG@10) |
|---|---|
| Tree depth 1, 2, 3, 4, 6, 8 at learning rate 0.1 | 0.0391, 0.0392, 0.0395, 0.0389, 0.0376, 0.0369 |
| Learning rate 0.05, 0.1, 0.2 at depth 3 | 0.0392, 0.0395, 0.0388 |
| Candidates reordered 200, 500, 1,000 | 0.0390, 0.0395, 0.0391 |
| Objective LambdaRank, yes/no | 0.0395, 0.0396 |

Chosen: 500 candidates, depth 3, learning rate 0.1, LambdaRank, 250 trees. Each of the
three tuned values has a worse one on either side.
Trees deeper than 4 do worse, and worse the longer they train. With 0.87% of rows
positive, they fit the quirks of one 28-day period.
Cost: settings in different stages are never varied together.
Caveat: depths 1 to 4 are within 0.0006 of each other, and a single run moves by up to
0.0003 between neighbouring tree counts. That deep trees lose is a finding; the choice
among the shallow ones is not.

**LambdaRank is kept, although a plain yes/no objective does as well.**
At the chosen setting `binary:logistic` scores NDCG@10 = 0.0396 against 0.0395, so it wins
under the selection rule by 0.0001. The ranking objective adds nothing measurable to
NDCG@10 here.
Why LambdaRank all the same: the margin is noise, it is ahead on Recall@10 (0.0340
against 0.0335) and Recall@20 (0.0569 against 0.0557), and its top 10s reach more of the
catalogue (33.7% against 29.9%), which matters for the Phase 6 popularity analysis.
The yes/no model is the steadier of the two: its NDCG@10 stays between 0.0391 and 0.0396
from 75 to 600 trees.

**SHAP values come from XGBoost itself, not the `shap` package.**
`predict(pred_contribs=True)` is exact TreeSHAP, and a test checks the contributions add
up to the score.
Why: the package brings in pandas, scikit-learn and numba for a result XGBoost already
gives.

**Differences between two models come with a 95% interval from a paired bootstrap over users.**
`evaluate.difference` resamples users 2,000 times and takes the 2.5th and 97.5th
percentiles of the mean difference. Paired, because both models are scored on the same
users and most of the spread between users is shared.
Why: Phase 4 ended with two models 0.001 apart and no way to say whether that was noise.

**XGBoost is kept out of any process that loads PyTorch.**
Loaded after PyTorch it crashes (two OpenMP runtimes, as with FAISS). `ranker.py` imports
nothing that imports `torch` and reads other models' output from files:
`data/processed/recs/` and `data/processed/ranker/`. Its tests are in `tests/isolated/`
and run in a process of their own.
XGBoost needs the OpenMP library `libomp`; here it finds the copy that comes with
Anaconda's Python.

**Leakage checks.**
- A test builds the `train` snapshot with three different label periods and checks that
  every feature is identical.
- `candidates` refuses to build a snapshot unless the saved model's users and tracks are
  those of the snapshot's history, so a model fitted on later data cannot be used by mistake.
- A ranker trained on labels shuffled within each user scores NDCG@10 = 0.0089, against
  0.0349 for retrieval order: with no real labels there is nothing for it to find.
- The first real run gained 7.7% on NDCG@10 and the tuned model 13.2%, far from the
  doubling that would have triggered an audit. The largest SHAP values belong to the
  retrieval scores and to similarity with recent listening.

**Finding: the ranker leans on the model scores first, then on what the user played recently.**
The chosen setting retrained without one feature group at a time, on `val`. The full
model scores NDCG@10 = 0.0395 and retrieval order 0.0349.

| Without | NDCG@10 | Change | Share of SHAP in the full model |
|---|---|---|---|
| models | 0.0336 | -14.8% | 48% |
| pair | 0.0360 | -8.8% | 31% |
| track | 0.0381 | -3.4% | 17% |
| user | 0.0388 | -1.8% | 3% |

Without the model scores the ranker falls below retrieval order: the other features
cannot rebuild what the retrieval model knows. Without the pair features about three
quarters of the gain is gone. The largest single features are the `tt_id` score,
`recent_affinity` (similarity to the last 28 days of listening) and
`track_recent_share` (how much of a track's audience is recent).
User features count for little. A user feature is the same for all of a user's
candidates, so it can change an order only in combination with another feature.

**`test` was scored once, after the owner confirmed the setting and LambdaRank on 2026-10-10.**
`tune`, `ablate` and `val` read `val` only.
Result on `test`, 10,280 users, every model fitted on both train splits:

| Model | Recall@10 | NDCG@10 | Recall@100 | Recall@500 | Coverage@10 |
|---|---|---|---|---|---|
| Popularity | 0.0042 | 0.0061 | 0.0292 | 0.0950 | 0.1% |
| Item-kNN | 0.0208 | 0.0276 | 0.1088 | 0.2841 | 19.7% |
| ALS | 0.0244 | 0.0284 | 0.1356 | 0.3434 | 22.1% |
| `tt_id` | 0.0278 | 0.0325 | 0.1468 | 0.3579 | 28.7% |
| `tt_id` + ranker | 0.0310 | 0.0348 | 0.1508 | 0.3579 | 33.5% |

The ranker against `tt_id` alone, with 95% intervals:

| Metric | Difference | Interval | Relative |
|---|---|---|---|
| NDCG@10 | +0.0023 | +0.0010 to +0.0035 | +7.0% |
| Recall@10 | +0.0032 | +0.0016 to +0.0046 | +11.3% |
| Recall@100 | +0.0040 | +0.0015 to +0.0066 | +2.7% |

Every interval is clear of zero. The gain in NDCG@10 is about half of the 13.2% on `val`,
as expected when the setting and the tree count are picked on `val`.
Recall@500 is unchanged by construction: the ranker reorders the first 500.
Against ALS the two-stage system is 22% ahead on NDCG@10 and 27% on Recall@10.
Fresher data alone is worth 5 to 7% on NDCG@10 on `test` for item-kNN, ALS and `tt_id`.
For Phase 6: the trained ranker and its feature list are in
`data/processed/models/ranker/`. The Music4All metadata arrived on 2026-10-10, so artist
features can now be tried.
