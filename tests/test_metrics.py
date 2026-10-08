import numpy as np
import pytest

from stsm_integration import metrics as M

rng = np.random.default_rng(0)


def grid(n=20):
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    return np.c_[xs.ravel(), ys.ravel()].astype(float)


def test_agreement_perfect_and_relabel_invariant():
    ref = np.repeat(["a", "b", "c"], 30)
    pred = np.repeat([2, 0, 1], 30)
    r = M.cluster_agreement(ref, pred)
    assert r["ARI"] == pytest.approx(1) and r["NMI"] == pytest.approx(1)


def test_agreement_ignores_missing_ref():
    ref = np.array(["a"] * 10 + ["b"] * 10 + [np.nan] * 5, dtype=object)
    pred = np.r_[np.zeros(10), np.ones(10), rng.integers(0, 2, 5)]
    r = M.cluster_agreement(ref, pred)
    assert r["n_scored"] == 20 and r["ARI"] == pytest.approx(1)


def test_silhouette_uses_reference():
    emb = np.r_[rng.normal(0, .1, (50, 2)), rng.normal(5, .1, (50, 2))]
    good = np.repeat([0, 1], 50)
    bad = rng.permutation(good)
    assert M.silhouette_vs_reference(emb, good) > 0.9
    assert M.silhouette_vs_reference(emb, bad) < 0.1


def test_morans_smooth_vs_random():
    c = grid()
    smooth = c[:, :1]
    noise = rng.normal(size=(len(c), 1))
    assert M.morans_i(c, smooth)[0] > 0.8
    assert abs(M.morans_i(c, noise)[0]) < 0.1


def test_ilisi_mixed_vs_separated():
    a, b = rng.normal(0, 1, (200, 5)), rng.normal(0, 1, (200, 5))
    mod = np.r_[np.zeros(200), np.ones(200)]
    mixed = M.ilisi(np.vstack([a, b]), mod)
    sep = M.ilisi(np.vstack([a, b + 50]), mod)
    assert mixed["iLISI"] > 1.8 and mixed["iLISI_norm"] > 0.8
    assert sep["iLISI"] < 1.05 and sep["iLISI_norm"] < 0.05


def test_ilisi_unbalanced_counts_still_high_when_mixed():
    a, b = rng.normal(0, 1, (400, 5)), rng.normal(0, 1, (100, 5))
    mod = np.r_[np.zeros(400), np.ones(100)]
    # standard LISI is bounded by composition: 400/100 mix gives ~1.6, not 2
    assert M.ilisi(np.vstack([a, b]), mod)["iLISI"] > 1.4


def test_preservation_identity_and_random():
    x = rng.normal(size=(300, 10))
    assert M.preservation(x, x) == pytest.approx(1)
    assert abs(M.preservation(x, rng.normal(size=(300, 10)))) < 0.1


def test_score_embedding_with_given_labels():
    z = np.r_[rng.normal(0, .1, (60, 4)), rng.normal(3, .1, (60, 4))]
    ref = np.repeat(["x", "y"], 60)
    out = M.score_embedding(z, z + rng.normal(0, .05, z.shape), ref,
                            coords_st=rng.uniform(size=(120, 2)), x_st=z, pred_st=ref)
    assert out["ARI"] == pytest.approx(1)
    assert out["iLISI"] > 1.8
    assert out["ST_preservation"] == pytest.approx(1)


def test_leiden_to_k():
    pytest.importorskip("scanpy"); pytest.importorskip("igraph")
    z = np.r_[[rng.normal(i * 10, .5, (40, 3)) for i in range(4)]].reshape(-1, 3)
    labels, n = M.leiden_to_k(z, 4)
    assert n == 4