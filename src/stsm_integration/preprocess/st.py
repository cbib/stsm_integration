"""Spatial transcriptomics preprocessing (Visium). One slice in, one slice out."""
from __future__ import annotations

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from scipy.stats import median_abs_deviation

VISIUM_SPOT_UM = 55.0 #physical spot (metadata only)
SPACERANGER_REF_UM = 65.0 # what spot_diameter_fullres measures

def load_visium(path: str, slice_id: str) -> AnnData:
    """Read a Space Ranger `outs/` folder; keep raw counts in layers['counts']."""
    import squidpy as sq

    adata = sq.read.visium(path)
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    adata.uns["slice_id"] = slice_id
    return adata


def coords_to_um(adata: AnnData, ref_um: float = SPACERANGER_REF_UM) -> None:
    """Convert obsm['spatial'] from full-res pixels to µm; keep pixels in obsm['spatial_px']."""
    lib = next(iter(adata.uns["spatial"]))
    diam_px = adata.uns["spatial"][lib]["scalefactors"]["spot_diameter_fullres"]
    um_per_px = ref_um / diam_px
    adata.obsm["spatial_px"] = np.asarray(adata.obsm["spatial"], dtype=float)
    adata.obsm["spatial"] = adata.obsm["spatial_px"] * um_per_px
    adata.uns["um_per_px"] = um_per_px


def mito_cutoff(pct: np.ndarray, n_mad: float = 3.0, floor: float = 5.0) -> float:
    """Cutoff = max(median + n_mad·MAD, floor), in percent."""
    return max(float(np.median(pct) + n_mad * median_abs_deviation(pct)), floor)


def qc_filter(adata, mito_prefix="mt-", min_counts=500, min_genes=250,
              min_cells=1, n_mad=3.0, mito_mode="flag"):
    """QC one slice.

    Removes spots below absolute floors (technical failure).
    Flags, never removes: obs['low_quality'] (below median − n_mad·MAD on log1p
    counts or genes) and obs['high_mito'] (unless mito_mode='filter').
    """
    adata.var["mt"] = adata.var_names.str.lower().str.startswith(mito_prefix.lower())
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], layer="counts", percent_top=None, inplace=True)

    def low_bound(x):
        lx = np.log1p(x)
        return float(np.expm1(np.median(lx) - n_mad * median_abs_deviation(lx)))

    adapt_c = low_bound(adata.obs["total_counts"])
    adapt_g = low_bound(adata.obs["n_genes_by_counts"])
    adata.obs["low_quality"] = (adata.obs["total_counts"] < adapt_c) | (adata.obs["n_genes_by_counts"] < adapt_g)
    cut = mito_cutoff(adata.obs["pct_counts_mt"].to_numpy(), n_mad)
    adata.obs["high_mito"] = adata.obs["pct_counts_mt"] >= cut

    keep = (adata.obs["total_counts"] >= min_counts) & (adata.obs["n_genes_by_counts"] >= min_genes)
    if mito_mode == "filter":
        keep &= ~adata.obs["high_mito"]
    n0, g0 = adata.n_obs, adata.n_vars
    adata = adata[keep.to_numpy()].copy()
    gkeep = np.asarray((adata.layers["counts"] > 0).sum(axis=0)).ravel() >= min_cells
    adata = adata[:, gkeep].copy()
    adata.uns["qc"] = {"spots_in": n0, "spots_out": adata.n_obs, "genes_in": g0, "genes_out": adata.n_vars,
                       "min_counts": min_counts, "min_genes": min_genes,
                       "adaptive_counts": adapt_c, "adaptive_genes": adapt_g,
                       "n_low_quality_kept": int(adata.obs["low_quality"].sum()),
                       "mito_cutoff_pct": cut, "mito_mode": mito_mode,
                       "n_high_mito_kept": int(adata.obs["high_mito"].sum())}
    return adata


def lognormalize(adata: AnnData, target_sum: float = 1e4) -> None:
    """layers['lognorm'] = log1p(normalize_total(counts)); X = lognorm. Never scales."""
    x = adata.layers["counts"].astype(np.float32)
    totals = np.asarray(x.sum(axis=1)).ravel()
    totals[totals == 0] = 1
    x = (sp.diags(target_sum / totals) @ x) if sp.issparse(x) else x * (target_sum / totals)[:, None]
    x = x.log1p() if sp.issparse(x) else np.log1p(x)
    adata.layers["lognorm"] = sp.csr_matrix(x) if sp.issparse(adata.layers["counts"]) else x
    adata.X = adata.layers["lognorm"].copy()


def preprocess_st(adata: AnnData, cfg: dict) -> AnnData:
    """Full ST pipeline for one slice. cfg: the 'st' block of a cohort config."""
    if "spatial" in adata.uns:
        coords_to_um(adata)
    adata = qc_filter(
        adata,
        mito_prefix=cfg.get("mito_prefix", "mt-"),
        min_counts=cfg.get("min_counts", None),
        min_genes=cfg.get("min_genes", None),
        min_cells=cfg.get("min_cells", 1),
        mito_mode=cfg.get("mito_mode", "flag"),
    )
    lognormalize(adata)
    adata.uns["modality"] = "st"
    adata.uns["pixel_size_um"] = cfg.get("spot_diameter_um", VISIUM_SPOT_UM)
    return adata