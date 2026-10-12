import numpy as np
import pytest
import torch
import torch.nn.functional as F

from krec.retrieval.model import TwoTower, in_batch_softmax_loss

ITEM_SIZES = {"author_id": 9, "duration": 5, "tag": 7}
USER_SIZES = {"user_active_degree": 4, "onehot_feat0": 3}


def _item_batch(b, n=None, t=2, seed=0, low=1):
    g = torch.Generator().manual_seed(seed)
    shape = (b,) if n is None else (b, n)
    fields = {
        f: torch.randint(low, size, shape, generator=g)
        for f, size in ITEM_SIZES.items()
        if f != "tag"
    }
    tags = torch.randint(low, ITEM_SIZES["tag"], (*shape, t), generator=g)
    return fields, tags


def _profile(b):
    return {f: torch.ones(b, dtype=torch.long) for f in USER_SIZES}


def test_loss_matches_a_manual_softmax():
    rng = np.random.default_rng(0)
    u = F.normalize(torch.tensor(rng.normal(size=(4, 3))), dim=1)
    v = F.normalize(torch.tensor(rng.normal(size=(4, 3))), dim=1)
    ids, log_q = torch.tensor([0, 1, 2, 3]), torch.log(
        torch.tensor([0.1, 0.2, 0.3, 0.4])
    )
    logits = (u @ v.T).numpy() / 0.5 - log_q.numpy()[None, :]
    want = np.mean([np.log(np.exp(logits[i]).sum()) - logits[i, i] for i in range(4)])
    got = in_batch_softmax_loss(u, v, ids, log_q, 0.5).item()
    assert got == pytest.approx(want)


def test_same_video_in_another_row_is_not_a_negative():
    rng = np.random.default_rng(1)
    u, v = rng.normal(size=(3, 2)), rng.normal(size=(3, 2))
    ids = np.array([5, 5, 6])
    logits = u @ v.T
    # rows 0 and 1 share video 5, so neither counts as the other's negative
    allowed = (ids[:, None] != ids[None, :]) | np.eye(3, dtype=bool)
    want = np.mean(
        [np.log(np.exp(logits[i][allowed[i]]).sum()) - logits[i, i] for i in range(3)]
    )
    got = in_batch_softmax_loss(
        torch.tensor(u), torch.tensor(v), torch.tensor(ids), torch.zeros(3), 1.0
    )
    assert got.item() == pytest.approx(want)


def test_uniform_log_q_is_a_no_op():
    rng = np.random.default_rng(2)
    u, v = torch.tensor(rng.normal(size=(5, 4))), torch.tensor(rng.normal(size=(5, 4)))
    ids = torch.arange(5)
    plain = in_batch_softmax_loss(u, v, ids, torch.zeros(5), 0.2)
    uniform = in_batch_softmax_loss(u, v, ids, torch.full((5,), np.log(0.2)), 0.2)
    assert plain.item() == pytest.approx(uniform.item())


def test_towers_output_unit_vectors_and_handle_empty_histories():
    torch.manual_seed(0)
    m = TwoTower(ITEM_SIZES, USER_SIZES, field_dim=4, user_field_dim=2, hidden=8, dim=6)
    fields, tags = _item_batch(3)
    v = m.item_vectors(fields, tags)
    hf, ht = _item_batch(3, n=5, low=0)
    hf = {f: torch.zeros_like(x) for f, x in hf.items()}  # all slots empty
    u = m.user_vectors(hf, torch.zeros_like(ht), _profile(3))
    assert v.shape == (3, 6) and u.shape == (3, 6)
    assert torch.allclose(v.norm(dim=1), torch.ones(3))
    assert torch.isfinite(u).all()


def test_pooled_history_equals_the_mean_of_video_embeddings():
    torch.manual_seed(0)
    m = TwoTower(ITEM_SIZES, USER_SIZES, field_dim=4)
    hf, ht = _item_batch(2, n=3, t=2)  # every slot and tag known (index >= 1)
    pooled = m.content.pooled(hf, ht)
    per_video = m.content.items(
        {f: x.flatten() for f, x in hf.items()}, ht.flatten(0, 1)
    ).view(2, 3, -1)
    assert torch.allclose(pooled, per_video.mean(dim=1), atol=1e-6)


def test_user_tower_trains_the_shared_content_tables():
    m = TwoTower(ITEM_SIZES, USER_SIZES, field_dim=4)
    hf, ht = _item_batch(2, n=3)
    m.user_vectors(hf, ht, _profile(2)).sum().backward()
    grad = m.content.fields.table.weight.grad
    assert grad.is_sparse and grad.coalesce().values().abs().sum() > 0
    sparse, dense = m.parameter_groups()
    assert all(p.grad is None or p.grad.is_sparse for p in sparse)
    assert len(sparse) + len(dense) == len(list(m.parameters()))


def test_field_table_gives_each_field_its_own_rows():
    from krec.retrieval.model import FieldTable

    t = FieldTable({"a": 3, "b": 4}, dim=2)  # a: 1..2, b: 1..3, 0 is padding
    assert t.table.num_embeddings == 1 + 2 + 3
    a = t.rows("a", torch.tensor([0, 1, 2]))
    b = t.rows("b", torch.tensor([0, 1, 2, 3]))
    assert a.tolist() == [0, 1, 2] and b.tolist() == [0, 3, 4, 5]


def test_history_can_read_the_tables_without_training_them():
    m = TwoTower(ITEM_SIZES, USER_SIZES, field_dim=4, history_grad=False)
    hf, ht = _item_batch(2, n=3)
    m.user_vectors(hf, ht, _profile(2)).sum().backward()
    assert m.content.fields.table.weight.grad is None
    assert m.profile.table.weight.grad is not None
