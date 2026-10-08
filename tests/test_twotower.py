"""What the owner's three functions in twotower.py must return, on numbers worked out by hand."""
import math

import numpy as np
import pytest
import torch

from m4a_rec.twotower import TrackTower, UserTower, in_batch_softmax_loss

E = math.e


def set_weights(layer, rows):
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(rows, dtype=torch.float32))
        if getattr(layer, "bias", None) is not None:
            layer.bias.zero_()


def rows(out, expected):
    """True if the tensor matches the hand-written rows."""
    return out.detach().numpy() == pytest.approx(np.array(expected, dtype=float))


def content_tower(use_id):
    """Two tracks, two features each, and a content network that returns relu(features)."""
    features = torch.tensor([[0.0, 4.0], [-5.0, 2.0]])
    tower = TrackTower(n_tracks=2, dim=2, use_id=use_id, features=features, hidden=2).eval()
    set_weights(tower.content[0], [[1, 0], [0, 1]])
    set_weights(tower.content[3], [[1, 0], [0, 1]])
    return tower


# ---- towers


def test_user_tower_returns_unit_rows():
    tower = UserTower(n_users=3, dim=2)
    set_weights(tower.embedding, [[3, 4], [0, 2], [-1, 0]])
    out = tower(torch.tensor([0, 2, 0]))
    assert out.shape == (3, 2)
    assert rows(out, [[0.6, 0.8], [-1, 0], [0.6, 0.8]])  # [3, 4] has length 5


def test_track_tower_id_only():
    tower = TrackTower(n_tracks=2, dim=2, use_id=True)
    set_weights(tower.embedding, [[3, 4], [0, 2]])
    assert tower.content is None
    assert rows(tower(torch.tensor([1, 0])), [[0, 1], [0.6, 0.8]])


def test_track_tower_content_only():
    tower = content_tower(use_id=False)
    assert tower.embedding is None
    # track 0: relu([0, 4]) = [0, 4] -> [0, 1].  track 1: relu([-5, 2]) = [0, 2] -> [0, 1]
    assert rows(tower(torch.tensor([0, 1])), [[0, 1], [0, 1]])


def test_track_tower_hybrid_adds_the_two_parts_before_scaling():
    tower = content_tower(use_id=True)
    set_weights(tower.embedding, [[3, 0], [4, 1]])
    # track 0: [3, 0] + [0, 4] = [3, 4] -> [0.6, 0.8].  track 1: [4, 1] + [0, 2] = [4, 3] -> [0.8, 0.6]
    assert rows(tower(torch.tensor([0, 1])), [[0.6, 0.8], [0.8, 0.6]])


def test_towers_pass_gradients_to_their_weights():
    user, track = UserTower(4, 3), content_tower(use_id=True)
    (user(torch.tensor([1, 2])).sum() + track(torch.tensor([0, 1])).sum()).backward()
    assert user.embedding.weight.grad.abs().sum() > 0
    assert track.embedding.weight.grad.abs().sum() > 0
    assert track.content[0].weight.grad.abs().sum() > 0
    assert not track.features.requires_grad  # the features themselves are not trained


# ---- loss, stage 1

EYE2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # user i lines up with track i and not the other


def test_loss_two_pairs_hand_checked():
    # scores = [[1, 0], [0, 1]]; each row: -log(e / (e + 1)) = log(1 + 1/e) = 0.3133
    loss = in_batch_softmax_loss(EYE2, EYE2, temperature=1.0)
    assert loss.shape == ()
    assert loss.item() == pytest.approx(math.log(1 + 1 / E))
    assert loss.item() == pytest.approx(0.3133, abs=1e-4)


def test_loss_temperature_divides_the_scores():
    # scores = [[2, 0], [0, 2]]; each row: log(1 + e^-2) = 0.1269
    assert in_batch_softmax_loss(EYE2, EYE2, temperature=0.5).item() == pytest.approx(math.log(1 + E**-2))


def test_loss_is_log_batch_size_when_every_score_is_equal():
    same = torch.tensor([[1.0, 0.0]] * 4)
    assert in_batch_softmax_loss(same, same, temperature=0.1).item() == pytest.approx(math.log(4))


def test_loss_is_lower_when_users_line_up_with_their_own_track():
    swapped = EYE2.flip(0)  # user 0 now lines up with track 1
    right = in_batch_softmax_loss(EYE2, EYE2, temperature=1.0)
    wrong = in_batch_softmax_loss(EYE2, swapped, temperature=1.0)
    assert right < wrong
    assert wrong.item() == pytest.approx(math.log(1 + E))  # each row: -log(1 / (1 + e))


def test_loss_passes_gradients_to_both_towers():
    u = torch.randn(5, 3, requires_grad=True)
    v = torch.randn(5, 3, requires_grad=True)
    in_batch_softmax_loss(u, v, temperature=0.2).backward()
    assert u.grad.abs().sum() > 0 and v.grad.abs().sum() > 0


# ---- loss, stage 2


def test_loss_log_q_hand_checked():
    # Track 0 is three times as common as track 1: q = [0.75, 0.25].
    # scores - log q = [[1 - log .75, 0 - log .25], [0 - log .75, 1 - log .25]]
    # row 0: log(1 + e^((0 - log .25) - (1 - log .75))) = log(1 + 3/e) = 0.7436
    # row 1: log(1 + e^((0 - log .75) - (1 - log .25))) = log(1 + 1/(3e)) = 0.1157
    log_q = torch.tensor([0.75, 0.25]).log()
    loss = in_batch_softmax_loss(EYE2, EYE2, temperature=1.0, log_q=log_q)
    assert loss.item() == pytest.approx((math.log(1 + 3 / E) + math.log(1 + 1 / (3 * E))) / 2)
    assert loss.item() == pytest.approx(0.4296, abs=1e-4)


def test_loss_ignores_a_wrong_answer_that_is_the_same_track():
    # Rows 0 and 1 are the same track (7); row 2 is track 9.
    vec = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    track_idx = torch.tensor([7, 7, 9])
    # scores = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
    # unmasked: rows 0 and 1: -log(e / (2e + 1)) = log(2 + 1/e); row 2: log(1 + 2/e)
    plain = in_batch_softmax_loss(vec, vec, temperature=1.0)
    assert plain.item() == pytest.approx((2 * math.log(2 + 1 / E) + math.log(1 + 2 / E)) / 3)
    # masked: rows 0 and 1 no longer see each other: log(1 + 1/e); row 2 is unchanged
    masked = in_batch_softmax_loss(vec, vec, temperature=1.0, track_idx=track_idx)
    assert masked.item() == pytest.approx((2 * math.log(1 + 1 / E) + math.log(1 + 2 / E)) / 3)


def test_loss_corrections_leave_a_batch_without_duplicates_or_skew_unchanged():
    u, v = torch.randn(6, 4), torch.randn(6, 4)
    plain = in_batch_softmax_loss(u, v, temperature=0.3)
    no_duplicates = in_batch_softmax_loss(u, v, temperature=0.3, track_idx=torch.arange(6))
    even = in_batch_softmax_loss(u, v, temperature=0.3, log_q=torch.full((6,), math.log(1 / 6)))
    assert no_duplicates.item() == pytest.approx(plain.item(), rel=1e-5)
    assert even.item() == pytest.approx(plain.item(), rel=1e-5)  # the same shift in every column cancels
