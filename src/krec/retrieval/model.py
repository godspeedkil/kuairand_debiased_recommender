"""
Two-tower model: a video tower over content and a user tower over the user's
history and profile, trained with an in-batch softmax.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FieldTable(nn.Module):
    """
    One embedding table holding every field, each in its own block of rows.
    One table instead of one per field for one sparse gradient, and a single
    optimizer update per step.

    Parameters
    ----------
    sizes : dict[str, int]
        Vocabulary size per field, index 0 included.
    dim : int
        Embedding size.
    """

    def __init__(self, sizes: dict[str, int], dim: int):
        super().__init__()
        self.names = sorted(sizes)
        start, self.base = 1, {}
        for f in self.names:
            self.base[f] = start
            start += sizes[f] - 1
        self.table = nn.Embedding(start, dim, padding_idx=0, sparse=True)

    def rows(self, field: str, idx: torch.Tensor) -> torch.Tensor:
        """
        Table rows for a field's indices.

        Parameters
        ----------
        field : str
            Field name.
        idx : torch.Tensor
            Indices within the field.

        Returns
        -------
        torch.Tensor
            Indices into the shared table, same shape.
        """
        return torch.where(idx > 0, idx + (self.base[field] - 1), 0)

    def lookup(self, fields: list[str], idx: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Embeddings of several fields in one table call.

        Parameters
        ----------
        fields : list[str]
            Fields to look up, in output order.
        idx : dict[str, torch.Tensor]
            Indices per field, shape (B,).

        Returns
        -------
        torch.Tensor
            Shape (B, len(fields) * dim).
        """
        rows = torch.stack([self.rows(f, idx[f]) for f in fields], dim=1)
        return self.table(rows).flatten(1)

    def mean(self, fields: list[str], idx: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Per-field mean embedding over the non-padding entries.

        Parameters
        ----------
        fields : list[str]
            Fields to average, in output order.
        idx : dict[str, torch.Tensor]
            Indices per field, shape (B, M).

        Returns
        -------
        torch.Tensor
            Shape (B, len(fields) * dim); zeros where a row has only padding.
        """
        rows = torch.stack([self.rows(f, idx[f]) for f in fields], dim=1)
        b, n_fields, m = rows.shape
        # one bag per (row, field); a 2-D input makes each of its rows a bag
        out = F.embedding_bag(
            rows.reshape(b * n_fields, m),
            self.table.weight,
            mode="mean",
            padding_idx=0,
            sparse=True,
        )
        return out.view(b, -1)


class ContentEmbedding(nn.Module):
    """
    Embeddings for video content fields, shared by both towers.

    Parameters
    ----------
    sizes : dict[str, int]
        Vocabulary size per field.
    dim : int
        Embedding size per field.
    """

    def __init__(self, sizes: dict[str, int], dim: int):
        super().__init__()
        self.fields = FieldTable(sizes, dim)
        self.single = [f for f in self.fields.names if f != "tag"]
        self.out_dim = dim * len(self.fields.names)

    def items(
        self, fields: dict[str, torch.Tensor], tags: torch.Tensor
    ) -> torch.Tensor:
        """
        Field embeddings, then the mean tag embedding, one row per video.

        Parameters
        ----------
        fields : dict[str, torch.Tensor]
            Index per video for each single-valued field, shape (B,).
        tags : torch.Tensor
            Tag indices, shape (B, T), 0 for padding.

        Returns
        -------
        torch.Tensor
            Shape (B, out_dim).
        """
        single = self.fields.lookup(self.single, fields)
        return torch.cat([single, self.fields.mean(["tag"], {"tag": tags})], dim=1)

    def pooled(
        self, fields: dict[str, torch.Tensor], tags: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean of each field's embeddings over a set of videos, per row.

        Parameters
        ----------
        fields : dict[str, torch.Tensor]
            Index per video for each single-valued field, shape (B, N).
        tags : torch.Tensor
            Tag indices, shape (B, N, T).

        Returns
        -------
        torch.Tensor
            Shape (B, out_dim), laid out like `items`; zeros where a row has no
            known values.
        """
        single = self.fields.mean(self.single, fields)
        tag = self.fields.mean(["tag"], {"tag": tags.flatten(1)})
        return torch.cat([single, tag], dim=1)


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))


class TwoTower(nn.Module):
    """
    Video and user towers mapping into a shared, L2-normalized space.

    Parameters
    ----------
    item_sizes : dict[str, int]
        Vocabulary size per video content field.
    user_sizes : dict[str, int]
        Vocabulary size per user profile field.
    field_dim : int, default 32
        Embedding size per video field.
    user_field_dim : int, default 8
        Embedding size per user profile field.
    hidden : int, default 256
        Hidden layer size of both towers.
    dim : int, default 64
        Output embedding size.
    history_grad : bool, default True
        Whether the user tower backpropagates into the history's embeddings.
    """

    def __init__(
        self,
        item_sizes: dict[str, int],
        user_sizes: dict[str, int],
        field_dim: int = 32,
        user_field_dim: int = 8,
        hidden: int = 256,
        dim: int = 64,
        history_grad: bool = True,
    ):
        super().__init__()
        self.history_grad = history_grad
        self.content = ContentEmbedding(item_sizes, field_dim)
        self.profile = FieldTable(user_sizes, user_field_dim)
        profile_dim = user_field_dim * len(user_sizes)
        self.item_tower = _mlp(self.content.out_dim, hidden, dim)
        self.user_tower = _mlp(self.content.out_dim + profile_dim, hidden, dim)

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """
        Embedding tables (sparse gradients) and tower weights (dense gradients).

        Returns
        -------
        tuple[list[nn.Parameter], list[nn.Parameter]]
            Sparse parameters for `torch.optim.SparseAdam`, then dense ones for
            `torch.optim.Adam`.
        """
        sparse = [*self.content.parameters(), *self.profile.parameters()]
        ids = {id(p) for p in sparse}
        return sparse, [p for p in self.parameters() if id(p) not in ids]

    def item_vectors(
        self, fields: dict[str, torch.Tensor], tags: torch.Tensor
    ) -> torch.Tensor:
        """
        Unit-norm video embeddings.

        Parameters
        ----------
        fields : dict[str, torch.Tensor]
            Index per video for each single-valued field, shape (B,).
        tags : torch.Tensor
            Tag indices, shape (B, T).

        Returns
        -------
        torch.Tensor
            Shape (B, dim).
        """
        return F.normalize(self.item_tower(self.content.items(fields, tags)), dim=1)

    def user_vectors(
        self,
        history_fields: dict[str, torch.Tensor],
        history_tags: torch.Tensor,
        profile: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Unit-norm user embeddings from history content and profile.

        Parameters
        ----------
        history_fields : dict[str, torch.Tensor]
            Content index per history video and field, shape (B, N), 0 for empty
            slots.
        history_tags : torch.Tensor
            Tag indices per history video, shape (B, N, T).
        profile : dict[str, torch.Tensor]
            Index per user for each profile field, shape (B,).

        Returns
        -------
        torch.Tensor
            Shape (B, dim).
        """
        if self.history_grad:
            hist = self.content.pooled(history_fields, history_tags)
        else:  # read the shared embeddings, train them from the video side only
            with torch.no_grad():
                hist = self.content.pooled(history_fields, history_tags)
        prof = self.profile.lookup(self.profile.names, profile)
        x = torch.cat([hist, prof], dim=1)
        return F.normalize(self.user_tower(x), dim=1)


def in_batch_softmax_loss(
    users: torch.Tensor,
    items: torch.Tensor,
    item_ids: torch.Tensor,
    log_q: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Softmax over the batch's videos, with the log-Q correction.

    Row i's positive is video i; the other videos in the batch are its negatives.
    Popular videos show up as negatives more often, so each logit is lowered by
    the log of how often its video is sampled. Other rows of the same video are
    masked out, since they are not negatives.

    Parameters
    ----------
    users : torch.Tensor
        User vectors, shape (B, d).
    items : torch.Tensor
        Positive video vectors, shape (B, d).
    item_ids : torch.Tensor
        Catalog position of each row's video, shape (B,).
    log_q : torch.Tensor
        Log sampling probability of each row's video, shape (B,).
    temperature : float
        Divides the dot products; lower means a sharper softmax.

    Returns
    -------
    torch.Tensor
        Mean cross-entropy, a scalar.
    """
    logits = users @ items.T / temperature - log_q[None, :]
    same = item_ids[:, None] == item_ids[None, :]
    diag = torch.eye(len(item_ids), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same & ~diag, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(item_ids), device=logits.device))
