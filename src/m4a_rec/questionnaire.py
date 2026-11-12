"""What a questionnaire is worth: the new-listener test with answers in place of listens.

The listeners `python -m m4a_rec.newcomers` held out are served from none of their
listens, only from the answers they would give to a short form. The answers are read off
their own train-split listens: their strongest genres, which third of the known listeners
they fall in by how mainstream their tracks are, and their most-played artists. A person
filling in a form would be less exact, so every row is an upper bound.

Genre answers go to look-alike listeners: the known listeners whose genre profiles are
closest to the answer, and the average of the two-tower vectors learned for them. Artist
answers become listens, each named artist's most-listened tracks played once, and from
there the service runs as for anyone new. The rows from real listens (serve.recent) are
the listens curve's, so a form can be placed on it.

It imports serve.py, so like that module it must not import torch.

  tune    the look-alikes' neighbour count and the tracks per artist and per genre, on
          val. Prints; writes nothing.
  report  every form on val and test at the chosen setting. Writes reports/questionnaire.md.

Usage: python -m m4a_rec.questionnaire tune | report
"""
from __future__ import annotations

import sys
from typing import NamedTuple

import numpy as np
import polars as pl
import scipy.sparse as sp

from . import bundle, serve
from .config import load_config
from .evaluate import PAIR, TRAIN, paired_difference
from .interactions import load_train
from .rank_features import pair_matrix, unit
from .reporting import stable_json, table

NO_LISTENS = pl.DataFrame(schema={"user_id": pl.Int64, "track_id": pl.String, "timestamp": pl.Datetime("ms")})


class Sheet(NamedTuple):
    """What the answers are read from: the listeners asked, and the known listeners they are matched with."""

    users: pl.Series  # the listeners asked
    listens: pl.DataFrame  # their listens
    profile: np.ndarray  # (asked, genres): their genre profiles
    band: np.ndarray  # the third of the known listeners each falls in by how mainstream their tracks are
    known_profile: np.ndarray  # the same two for the bundle's listeners, in its user order
    known_band: np.ndarray
    by_artist: pl.DataFrame  # most_listened, per artist code
    by_genre: pl.DataFrame  # most_listened, per genre column: a track is under the genre it has most weight on


def count(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def taste(played: sp.csr_matrix, genres: np.ndarray, track_pct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From a users x tracks matrix of who played what: each user's genre profile and how mainstream they are.

    The profile is their tracks' genre weights added up. Mainstream is the mean of their
    tracks' popularity percentiles, the ranker's user_mainstream; 0 for a user with no track.
    """
    return np.asarray(played @ genres), np.asarray(played @ track_pct) / np.maximum(played.getnnz(axis=1), 1)


def strongest(profile: np.ndarray, n: int) -> np.ndarray:
    """Each listener's n strongest genres as an answer: 1 for a named genre, 0 for the rest.

    A tie goes to the lower column. A genre with no weight is never named, so a listener
    with fewer than n genres names fewer.
    """
    out = np.zeros(profile.shape, dtype=np.float32)
    np.put_along_axis(out, np.argsort(-profile, axis=1, kind="stable")[:, :n], 1.0, axis=1)
    return out * (profile > 0)


def thirds(values: np.ndarray, known: np.ndarray) -> np.ndarray:
    """0, 1 or 2: the third of `known` each value falls in. A value on a cut point goes to the lower third."""
    return np.searchsorted(np.quantile(known, [1 / 3, 2 / 3]), values)


def look_alikes(
    answers: np.ndarray, known_profiles: np.ndarray, known_vectors: np.ndarray, k: int, bands: tuple[np.ndarray, np.ndarray] | None = None
) -> np.ndarray:
    """For each answer, the direction of the average vector of the k known listeners whose profiles are closest to it.

    Closest by cosine; a tie goes to the earlier known listener. `bands` is (each answer's
    band, each known listener's band): only known listeners in the answer's band count,
    and every band must hold at least k of them.
    """
    sim = unit(answers) @ unit(known_profiles).T
    if bands is not None:
        sim[bands[0][:, None] != bands[1][None, :]] = -np.inf
    top = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    return unit(known_vectors[top].mean(axis=1))


def favourites(history: pl.DataFrame, track_ids: pl.Series, artist: np.ndarray, n: int) -> pl.DataFrame:
    """Each listener's n most-played artists, by listens: user_id and group, the artist's code. A tie goes to the lower code."""
    code = pl.DataFrame({"track_id": track_ids, "group": artist.astype(np.int64)})
    plays = history.join(code, on="track_id").group_by("user_id", "group").len(name="plays")
    first = plays.sort(["user_id", "plays", "group"], descending=[False, True, False]).group_by("user_id", maintain_order=True).head(n)
    return first.select("user_id", "group")


def named(answers: np.ndarray, users: pl.Series) -> pl.DataFrame:
    """The genres each listener names in `answers`, as `favourites` gives artists: user_id and group, the genre's column."""
    rows, cols = np.nonzero(answers)
    return pl.DataFrame({"user_id": users.gather(rows), "group": cols.astype(np.int64)})


def most_listened(groups: np.ndarray, listeners: np.ndarray, track_ids: pl.Series) -> pl.DataFrame:
    """Each group's tracks from the most listeners down: group, track_id, place (1 is the first).

    `groups` gives each track's group (an artist, a genre); a track with a negative group
    is in none. A tie goes to the lower track_id.
    """
    tracks = pl.DataFrame({"group": groups.astype(np.int64), "track_id": track_ids, "listeners": listeners}).filter(pl.col("group") >= 0)
    tracks = tracks.sort(["group", "listeners", "track_id"], descending=[False, True, False])
    return tracks.select("group", "track_id", place=pl.int_range(1, pl.len() + 1).over("group"))


def seed_tracks(picks: pl.DataFrame, ranked: pl.DataFrame, per: int, now) -> pl.DataFrame:
    """Listens that stand in for answers: the first `per` tracks of each group a listener named, each played once at `now`.

    `picks` has user_id and group, `ranked` comes from most_listened.
    """
    seeds = picks.join(ranked.filter(pl.col("place") <= per), on="group").select(PAIR).unique(maintain_order=True)
    return seeds.with_columns(timestamp=pl.lit(now, dtype=pl.Datetime("ms")))


def vector_lists(b: bundle.Bundle, users: pl.Series, vectors: np.ndarray, seeds: pl.DataFrame = NO_LISTENS) -> pl.DataFrame:
    """Each listener's candidates in retrieval order, from two-tower vectors alone: user_id, track_id, score.

    The tracks in `seeds` are left out as played; with none, nothing is.
    """
    idx, scores = bundle.retrieve(b, seeds, users, vectors)
    out = pl.DataFrame(
        {
            "user_id": users.gather(np.repeat(np.arange(idx.shape[0]), idx.shape[1])),
            "track_id": b.track_ids.gather(idx.ravel().clip(0)),
            "score": scores.ravel(),
        }
    )
    return out.filter(pl.Series(idx.ravel() >= 0))


def read_sheet(b: bundle.Bundle, listens: pl.DataFrame, users: pl.Series) -> Sheet:
    """The sheet for `users`, whose listens are `listens`, against the listeners the bundle knows."""
    genres, track = b.side["genres"], b.pop["track"]
    played = lambda plays, ids: pair_matrix(plays.select(PAIR).unique(), ids, b.track_ids)  # noqa: E731
    profile, mainstream = taste(played(listens, users), genres, track["track_pct"])
    known_profile, known_mainstream = taste(played(bundle.listens(b, b.user_ids.to_list()), b.user_ids), genres, track["track_pct"])
    main_genre = np.where(genres.any(axis=1), genres.argmax(axis=1), -1)
    return Sheet(
        users=users,
        listens=listens,
        profile=profile,
        band=thirds(mainstream, known_mainstream),
        known_profile=known_profile,
        known_band=thirds(known_mainstream, known_mainstream),
        by_artist=most_listened(b.artist, track["track_listeners"], b.track_ids),
        by_genre=most_listened(main_genre, track["track_listeners"], b.track_ids),
    )


def lists(s: serve.Service, users: pl.Series, look: np.ndarray | None, seeds: pl.DataFrame | None) -> dict[str, pl.DataFrame]:
    """The lists a form's answers give: {"retrieval": the two-tower order} and, with seed tracks, {"both": the ranker's order}.

    `look` holds look-alike vectors, `seeds` the listens that stand in for answers; one
    may be None. With both, the two-tower vector is the direction of the look-alike
    vector plus the one folded in from the seeds. The ranker needs listens for its
    features, so without seeds there is no "both".
    """
    b = s.bundle
    if seeds is None:
        return {"retrieval": vector_lists(b, users, look)}
    tt_user, als_user = bundle.vectors(b, seeds, users, b.meta["fold_in"])
    if look is not None:
        tt_user = unit(tt_user + look)
    return {"retrieval": vector_lists(b, users, tt_user, seeds), "both": serve.recommend(s, seeds, users, tt_user, als_user)[0]}


def form(
    s: serve.Service,
    sheet: Sheet,
    genres: int | str | None = None,
    mainstream: bool = False,
    artists: int | None = None,
    neighbours: int | None = None,
    per_artist: int | None = None,
) -> dict[str, pl.DataFrame]:
    """`lists` for a form that asks for `genres` strongest genres ("all": the whole profile), the mainstream question and `artists` favourite artists."""
    b, look, seeds = s.bundle, None, None
    if genres is not None:
        answers = sheet.profile if genres == "all" else strongest(sheet.profile, genres)
        bands = (sheet.band, sheet.known_band) if mainstream else None
        look = look_alikes(answers, sheet.known_profile, b.tt_user, neighbours, bands)
    if artists is not None:
        seeds = seed_tracks(favourites(sheet.listens, b.track_ids, b.artist, artists), sheet.by_artist, per_artist, b.meta["now"])
    return lists(s, sheet.users, look, seeds)


def genre_seeds(s: serve.Service, sheet: Sheet, genres: int, per: int) -> dict[str, pl.DataFrame]:
    """`lists` for genre answers turned into listens instead: each named genre's `per` most-listened tracks."""
    picks = named(strongest(sheet.profile, genres), sheet.users)
    return lists(s, sheet.users, None, seed_tracks(picks, sheet.by_genre, per, s.bundle.meta["now"]))


def render(rows: dict, floor: dict, users: dict, info: dict, ks: list[int]) -> str:
    """The questionnaire report.

    rows   form -> {questions, stages, retrieval: {split: metrics}, both: {split: metrics}
           where the ranker ran, gap: {split: {metric: {diff, low, high}}}}; `stages` names
           the list the listener gets, `gap` is that list minus the `info["against"]` row's
    floor  {split: metrics} for the most popular tracks
    users  {held_out, <split>: listeners scored}
    """
    main, r10, r100 = f"ndcg@{ks[0]}", f"recall@{ks[0]}", f"recall@{ks[2]}"
    c = info["chosen"]
    L = [
        "# What a questionnaire is worth",
        "",
        "Generated by `python -m m4a_rec.questionnaire report`. Do not edit by hand; rerun instead.",
        "",
        f"The {users['held_out']:,} listeners of the new-listener test (`reports/serving.md`) were left out of tt_id,",
        "ALS and the population numbers. Here the service is shown none of their listens, only the",
        "answers they would give to a short form. The answers are read off their own train-split",
        "listens:",
        "",
        f"- genres: their N strongest genres out of Music4All-Onion's {info['genres']}. A listener's weight on a",
        "  genre is their tracks' weights on it, added up.",
        "- mainstream: one more question with three answers. The answer is the third of the known",
        "  listeners they fall in by how popular their tracks are.",
        "- artists: their N most-played artists.",
        "",
        "A person filling in a form would be less exact than their own listens, so every",
        "questionnaire row is an upper bound on what the form would give.",
        "",
        "How an answer becomes a list:",
        "",
        f"- Genre answers go to look-alike listeners: the {c['neighbours']} known listeners whose genre profiles are",
        "  closest to the answer (within the same third, when the mainstream question is asked). The",
        "  two-tower vectors learned for them are averaged, and the list is the retrieval order for",
        "  that vector. The ranker is not used: its features need listens.",
        f"- Artist answers become listens: each named artist's {count(c['tracks_per_artist'], 'most-listened track')}, played once. From",
        f"  there the service runs as for anyone new: fold-in ({info['fold_in']}), retrieval, then the ranker",
        f"  (`{info['ranker']}`). The tracks that stand in for listens are left out of the list.",
        "- The form that asks both adds the look-alike vector to the one folded in from the artists'",
        "  tracks, and the ranker works from those tracks.",
        f"- `as seed tracks` turns genre answers into listens instead, each genre's {count(c['tracks_per_genre'], 'most-listened track')}.",
        "  This is how the questionnaire was first planned; the row shows why it is not used.",
        "",
        "`Stages` names what made the row's list: `retrieval` is the two-tower order, `both` has the",
        "ranker on top. The last column is the two-tower order alone for every row. The recent-track",
        "rows and the whole history are those of `reports/listens_curve.md`: what real listens give.",
        "",
        f"`Against {info['against']}` is {main} minus that row's, with a 95% interval from resampling the",
        f"held-out listeners {info['n_boot']:,} times.",
        "",
        "The neighbour count and the tracks per artist and per genre were chosen on val",
        f"(`questionnaire tune`: the highest mean {main} over the forms that use them, the smaller",
        "number on a tie). Test was run once, at those. As in the listens curve, a list is scored",
        "after the tracks the listener has already played are dropped, and the ranker was trained on",
        "a snapshot that held these listeners' earlier listens, so the `both` rows are a little",
        "optimistic.",
        "",
    ]
    for split in floor:
        L += [f"## {split} ({users[split]:,} held-out listeners with a track to find)", ""]
        out = [["nothing: the most popular tracks", 0, "", floor[split][main], floor[split][r10], floor[split][r100], "", ""]]
        for name, row in rows.items():
            got, g = row[row["stages"]][split], row["gap"][split][main]
            out.append(
                [
                    name,
                    "" if row["questions"] is None else row["questions"],
                    row["stages"],
                    got[main],
                    got[r10],
                    got[r100],
                    "" if name == info["against"] else f"{g['diff']:+.4f} ({g['low']:+.4f} to {g['high']:+.4f})",
                    row["retrieval"][split][main],
                ]
            )
        L += table(["Form", "Questions", "Stages", main, r10, r100, f"Against {info['against']}", f"{main}, retrieval alone"], out)
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv not in (["tune"], ["report"]):
        print(__doc__)
        return 2
    cfg = load_config()
    q, c, ks, processed, reports = cfg["questionnaire"], cfg["serve"], cfg["metrics"]["ks"], cfg["paths"]["processed"], cfg["paths"]["reports"]
    say = lambda message: print(message, flush=True)  # noqa: E731
    held = np.load(processed / "models" / "tt_id_newcomers" / "held_out_users.npy").tolist()
    s = serve.load(processed / "serving_newcomers")
    theirs = load_train(processed / "splits", TRAIN).filter(pl.col("user_id").is_in(held))
    sheet = read_sheet(s.bundle, theirs, pl.Series("user_id", held, dtype=pl.Int64))
    metrics = [f"ndcg@{ks[0]}", f"recall@{ks[0]}", *(f"recall@{k}" for k in ks[2:])]
    mean = lambda users: {m: float(users[m].mean()) for m in metrics}  # noqa: E731
    both = q["combined"]
    genre_forms = {
        count(n, "genre") + (" + mainstream" if band else ""): {"genres": n, "mainstream": band} for n in q["genres"] for band in (False, True)
    }
    artist_forms = {count(n, "artist"): {"artists": n} for n in q["artists"]}
    say(f"{len(held):,} listeners asked, {int((~sheet.profile.any(axis=1)).sum())} of them with no genre to name")

    if argv == ["tune"]:
        grid = q["grid"]
        on_val = lambda recs: float(serve.per_user(recs, "val", held, ks[0])[metrics[0]].mean())  # noqa: E731
        whole = {"whole genre profile": {"genres": "all"}}
        rows = []
        for k in grid["neighbours"]:
            say(f"{k} neighbours")
            values = [on_val(form(s, sheet, **asked, neighbours=k)["retrieval"]) for asked in (genre_forms | whole).values()]
            rows.append([k, *values, float(np.mean(values[:-1]))])
        say("\n".join(table(["neighbours", *genre_forms, *whole, "mean of the forms"], rows)))
        rows = []
        for per in grid["tracks_per_artist"]:
            say(f"{per} tracks per artist")
            values = [on_val(form(s, sheet, **asked, per_artist=per)["both"]) for asked in artist_forms.values()]
            rows.append([per, *values, float(np.mean(values))])
        say("\n".join(table(["tracks per artist", *artist_forms, "mean of the forms"], rows)))
        rows = [[per, on_val(genre_seeds(s, sheet, both["genres"], per)["both"])] for per in grid["tracks_per_genre"]]
        say("\n".join(table(["tracks per genre", count(both["genres"], "genre") + " as seed tracks"], rows)))
        return 0

    ch, boot, splits, fold_in = q["chosen"], cfg["bootstrap"], ("val", "test"), s.bundle.meta["fold_in"]
    tuned = {"neighbours": ch["neighbours"], "per_artist": ch["tracks_per_artist"]}
    real = lambda listens: {stages: serve.lists_for(s, listens, held, fold_in, c["batch_users"], stages) for stages in ("retrieval", "both")}  # noqa: E731
    asked = {  # form -> (questions, its lists)
        **{name: (a["genres"] + a["mainstream"], form(s, sheet, **a, **tuned)) for name, a in genre_forms.items()},
        "whole genre profile": (None, form(s, sheet, genres="all", **tuned)),
        count(both["genres"], "genre") + " as seed tracks": (both["genres"], genre_seeds(s, sheet, both["genres"], ch["tracks_per_genre"])),
        **{name: (a["artists"], form(s, sheet, **a, **tuned)) for name, a in artist_forms.items()},
        f"{count(both['genres'], 'genre')} + mainstream + {count(both['artists'], 'artist')}": (
            both["genres"] + 1 + both["artists"],
            form(s, sheet, genres=both["genres"], mainstream=True, artists=both["artists"], **tuned),
        ),
        **{count(n, "recent track"): (None, real(serve.recent(theirs, n))) for n in q["reference_listens"]},
        "whole history": (None, real(theirs)),
    }
    against = count(q["reference_listens"][0], "recent track")
    scored = {}
    for name, (_, made) in asked.items():
        say(name)
        scored[name] = {stages: {split: serve.per_user(recs, split, held, ks[0]) for split in splits} for stages, recs in made.items()}
    base = scored[against]["both"]
    rows = {}
    for name, frames in scored.items():
        stages = "both" if "both" in frames else "retrieval"
        rows[name] = {
            "questions": asked[name][0],
            "stages": stages,
            **{st: {split: mean(frames[st][split]) for split in splits} for st in frames},
            "gap": {split: paired_difference(frames[stages][split], base[split], metrics, boot["n"], boot["seed"]) for split in splits},
        }
    popular = pl.read_parquet(processed / "recs" / "popularity.parquet").filter(pl.col("user_id").is_in(held))
    floor = {split: mean(serve.per_user(popular, split, held, ks[0])) for split in splits}
    users = {"held_out": len(held), **{split: base[split].height for split in splits}}
    info = {
        "fold_in": fold_in, "ranker": c["ranker"], "n_boot": boot["n"], "chosen": ch, "against": against, "genres": sheet.profile.shape[1]
    }
    (reports / "questionnaire.json").write_text(stable_json({"users": users, "chosen": ch, "popularity": floor, "forms": rows}))
    (reports / "questionnaire.md").write_text(render(rows, floor, users, info, ks))
    say(f"wrote {reports / 'questionnaire.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
