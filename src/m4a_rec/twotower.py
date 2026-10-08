"""The two towers and the loss of the retrieval model.

A tower turns an index into a vector of length 1. A user's score for a track is the dot
product of their two vectors, so it lies between -1 and 1.

The layers are defined here. The three functions marked "Yours to write" are the owner's:
UserTower.forward, TrackTower.forward and in_batch_softmax_loss. tests/test_twotower.py
says what each must return.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """One trainable vector per user."""

    def __init__(self, n_users: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_users, dim)
        # Rows start at about length 1. PyTorch's default is about sqrt(dim), which makes each
        # optimizer step a much smaller turn of the vector and training many times slower.
        nn.init.normal_(self.embedding.weight, std=dim**-0.5)

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        """Yours to write.

        user_idx  (B,) int64, row numbers into self.embedding
        returns   (B, dim) float32, each row scaled to length 1

        Look up each user's row, then scale it to length 1.
        Useful: torch.nn.functional.normalize.
        """
        return F.normalize(self.embedding(user_idx), dim=-1)


class TrackTower(nn.Module):
    """A vector per track, from its ID, its content features, or both.

    tt_id       use_id=True,  features=None     self.embedding only
    tt_hybrid   use_id=True,  features given    self.embedding and self.content
    tt_content  use_id=False, features given    self.content only; self.embedding is None
    """

    def __init__(
        self,
        n_tracks: int,
        dim: int,
        use_id: bool,
        features: torch.Tensor | None = None,
        hidden: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_tracks, dim) if use_id else None
        if use_id:
            nn.init.normal_(self.embedding.weight, std=dim**-0.5)  # as in UserTower
        if features is None:
            self.features, self.content = None, None
        else:
            # (n_tracks, F) float32. A buffer moves to the GPU with the module but is not trained.
            self.register_buffer("features", features)
            self.content = nn.Sequential(
                nn.Linear(features.shape[1], hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, dim),
            )

    def forward(self, track_idx: torch.Tensor) -> torch.Tensor:
        """Yours to write.

        track_idx  (B,) int64, row numbers into self.embedding and self.features
        returns    (B, dim) float32, each row scaled to length 1

        Build a (B, dim) vector from the parts this tower has:
          - if self.embedding is not None: the track's embedding row
          - if self.content is not None: self.content applied to the track's feature row
          - if it has both: their sum
        then scale each row to length 1.
        """
        vec = None
        if self.embedding is not None:
            vec = self.embedding(track_idx)
        if self.content is not None:
            content_vec = self.content(self.features[track_idx])
            vec = content_vec if vec is None else vec + content_vec
        return F.normalize(vec, dim=-1)


def in_batch_softmax_loss(user_vec, track_vec, temperature, log_q=None, track_idx=None):
    batch_size = user_vec.shape[0]

    # scores[i, j] = user i . track j, scaled
    scores = user_vec @ track_vec.T / temperature

    # Popularity correction: subtract log_q[j] from every score in column j
    if log_q is not None:
        scores = scores - log_q[None, :]

    # Duplicate masking: same track as the right answer, but not the diagonal itself
    if track_idx is not None:
        same_track = track_idx[:, None] == track_idx[None, :]
        off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(same_track & off_diagonal, float("-inf"))

    # Row i's right answer is option i
    targets = torch.arange(batch_size, device=scores.device)
    return F.cross_entropy(scores, targets)