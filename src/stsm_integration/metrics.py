"""Integration metrics. All functions take arrays, never AnnData side effects."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


# ---------- clustering agreement ----------

def _valid(ref) -> np.ndarray:
    s = pd.Series(ref, dtype=object)
    return (s.notna() & (s.astype(str).str.lower() != "nan")).to_numpy()


def cluster_agreement(ref, pred) -> dict[str, float]:
    """ARI, NMI, AMI, FMI, homogeneity of pred vs ref; spots with missing ref ignored."""
    ref, pred = np.asarray(ref, dtype=object), np.asarray(pred, dtype=object)
    m = _valid(ref)
    r, p = ref[m].astype(str), pred[m].astype(str)
    return {
        "ARI": adjusted_rand_score(r, p),
        "NMI": normalized_mutual_info_score(r, p),
        "AMI": adjusted_mutual_info_score(r, p),
        "FMI": fowlkes_mallows_score(r, p),
        "Homogeneity": homogeneity_score(r, p),
        "n_scored": int(m.sum()),
    }


def silhouette_vs_reference(emb: np.ndarray, ref, max_n: int = 10000, seed: int = 0) -> float:
    """Silhouette of emb using reference labels (not the embedding's own clusters)."""
    m = _valid(ref)
    x, y = np.asarray(emb)[m], np.asarray(ref, dtype=object)[m].astype(str)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(silhouette_score(x, y, sample_size=min(max_n, len(y)), random_state=seed))


# ---------- spatial ----------

def spatial_weights(coords: np.ndarray, k: int = 6) -> sp.csr_matrix:
    """Row-normalised kNN spatial weight matrix (no self-loops)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    a = nn.kneighbors_graph(coords, mode="connectivity").tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    rs = np.asarray(a.sum(axis=1)).ravel()
    rs[rs == 0] = 1
    return sp.diags(1 / rs) @ a


def morans_i(coords: np.ndarray, features: np.ndarray, k: int = 6) -> np.ndarray:
    """Moran's I per feature column (sparse, O(n·k·d))."""
    w = spatial_weights(coords, k)
    x = np.asarray(features, dtype=float)
    x = x - x.mean(axis=0)
    num = np.einsum("ij,ij->j", x, w @ x)
    den = (x ** 2).sum(axis=0)
    n, s0 = x.shape[0], w.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (n / s0) * num / den


# ---------- mixing ----------

def _lisi_one(dist: np.ndarray, labels: np.ndarray, n_cat: int, perplexity: float) -> float:
    # Gaussian kernel calibrated to target perplexity (Korsunsky et al. 2019)
    target, beta, lo, hi = np.log(perplexity), 1.0, -np.inf, np.inf
    for _ in range(50):
        p = np.exp(-dist * beta)
        sp_ = p.sum()
        if sp_ == 0:
            p, sp_ = np.ones_like(dist), len(dist)
        h = np.log(sp_) + beta * (dist * p).sum() / sp_
        if abs(h - target) < 1e-5:
            break
        if h > target:
            lo, beta = beta, beta * 2 if hi == np.inf else (beta + hi) / 2
        else:
            hi, beta = beta, beta / 2 if lo == -np.inf else (beta + lo) / 2
    p = p / p.sum()
    probs = np.bincount(labels, weights=p, minlength=n_cat)
    return 1.0 / np.sum(probs ** 2)


def lisi(emb: np.ndarray, labels, perplexity: float = 30, k: int | None = None) -> np.ndarray:
    """Per-cell LISI (standard formulation). Range [1, n_categories]."""
    codes = pd.Categorical(np.asarray(labels)).codes
    n_cat = codes.max() + 1
    k = k or min(int(3 * perplexity), len(codes) - 1)
    d, idx = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb)
    d, idx = d[:, 1:] ** 2, idx[:, 1:]
    return np.array([_lisi_one(d[i], codes[idx[i]], n_cat, perplexity) for i in range(len(codes))])


def ilisi(emb: np.ndarray, modality, **kw) -> dict[str, float]:
    """Modality-mixing iLISI: raw median and scib-style normalised value in [0, 1]."""
    vals = lisi(emb, modality, **kw)
    n_cat = len(np.unique(modality))
    med = float(np.median(vals))
    return {"iLISI": med, "iLISI_norm": (med - 1) / (n_cat - 1) if n_cat > 1 else np.nan}


# ---------- structure preservation ----------

def preservation(x_ref: np.ndarray, emb: np.ndarray, n_sample: int = 2000, seed: int = 0) -> float:
    """Spearman correlation of pairwise distances, input space vs embedding (subsampled)."""
    n = len(emb)
    idx = np.random.default_rng(seed).choice(n, size=min(n_sample, n), replace=False)
    a = pairwise_distances(np.asarray(x_ref)[idx])[np.triu_indices(len(idx), 1)]
    b = pairwise_distances(np.asarray(emb)[idx])[np.triu_indices(len(idx), 1)]
    return float(spearmanr(a, b).correlation)


# ---------- clustering helper ----------

def leiden_to_k(emb: np.ndarray, k: int, n_neighbors: int = 15, max_iter: int = 30, seed: int = 0):
    """Leiden labels with cluster count as close as possible to k. Returns (labels, n_found)."""
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(np.zeros((len(emb), 1)))
    a.obsm["e"] = np.asarray(emb)
    sc.pp.neighbors(a, use_rep="e", n_neighbors=n_neighbors, random_state=seed)
    lo, hi, best = 0.01, 5.0, None
    for _ in range(max_iter):
        res = (lo + hi) / 2
        sc.tl.leiden(a, resolution=res, key_added="l", random_state=seed, flavor="igraph", n_iterations=2)
        n = a.obs["l"].nunique()
        if best is None or abs(n - k) < abs(best[1] - k):
            best = (a.obs["l"].to_numpy().copy(), n)
        if n == k:
            break
        lo, hi = (res, hi) if n < k else (lo, res)
    return best


# ---------- aggregate ----------

def score_embedding(
    z_st: np.ndarray,
    z_sm: np.ndarray,
    ref_st,
    coords_st: np.ndarray,
    x_st: np.ndarray | None = None,
    pred_st=None,
) -> dict[str, float]:
    """All metrics for one slice. ST spots carry the reference; SM enters via mixing.

    pred_st: cluster labels on z_st; computed with leiden_to_k if None.
    x_st: ST input features (e.g. PCA) for ST_preservation.
    """
    out: dict[str, float] = {}
    if pred_st is None:
        k = len(pd.unique(np.asarray(ref_st, dtype=object)[_valid(ref_st)]))
        pred_st, n_found = leiden_to_k(z_st, k)
        out["k_target"], out["k_found"] = k, n_found
    out.update(cluster_agreement(ref_st, pred_st))
    out["Silhouette_ref"] = silhouette_vs_reference(z_st, ref_st)
    out["MoransI_latent"] = float(np.nanmean(morans_i(coords_st, z_st)))
    mod = np.r_[np.zeros(len(z_st), int), np.ones(len(z_sm), int)]
    out.update(ilisi(np.vstack([z_st, z_sm]), mod))
    if x_st is not None:
        out["ST_preservation"] = preservation(x_st, z_st)
    return out