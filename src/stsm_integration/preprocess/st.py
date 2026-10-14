"""Spatial transcriptomics preprocessing (Visium). One slice in, one slice out."""
from __future__ import annotations

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from scipy.stats import median_abs_deviation
import pandas as pd
import json
from pathlib import Path
from matplotlib.image import imread

VISIUM_SPOT_UM = 55.0 #physical spot (metadata only)
VISIUM_PITCH_UM = 100.0   # centre-to-centre, fixed by chip design
SPACERANGER_REF_UM = 65.0 # what spot_diameter_fullres measures

def load_visium(path: str, slice_id: str) -> AnnData:
    """Read a Space Ranger `outs/` folder; keep raw counts in layers['counts']."""
    import squidpy as sq

    adata = sq.read.visium(path)
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    adata.uns["slice_id"] = slice_id
    return annotate_species(adata)

SPECIES_BY_PREFIX = {"ENSMUSG": "mouse", "ENSG": "human"}   # mouse first: "ENSG" is a prefix of neither, but keep order explicit

def annotate_species(adata):
    """Set var['species'], uns['genome'] ('human'|'mouse'|'both'|'unknown'), obs['human_frac'] if both."""
    ids = adata.var.get("gene_ids", pd.Series("", index=adata.var_names)).astype(str)
    species = np.full(adata.n_vars, "unknown", dtype=object)
    species[ids.str.startswith("ENSMUSG").values] = "mouse"
    species[ids.str.startswith("ENSG").values] = "human"
    adata.var["species"] = pd.Categorical(species)
    found = set(species) - {"unknown"}
    adata.uns["genome"] = "both" if len(found) == 2 else (found.pop() if found else "unknown")
    if adata.uns["genome"] == "both":
        X = adata.layers["counts"]
        hum = np.asarray(X[:, species == "human"].sum(1)).ravel()
        tot = np.asarray(X.sum(1)).ravel()
        adata.obs["human_frac"] = np.divide(hum, tot, out=np.zeros_like(hum, float), where=tot > 0)
    return adata


def load_stq(path, slice_id):
    """Load an STQ output (raw mtx + spatial/) as a Visium-like AnnData, in-tissue spots only."""
    path = Path(path)
    adata = sc.read_10x_mtx(path / "raw_feature_bc_matrix", var_names="gene_symbols", gex_only=False)
    adata.var_names_make_unique()
    pos = pd.read_csv(path / "spatial/tissue_positions.csv").set_index("barcode")
    pos = pos.loc[pos["in_tissue"] == 1]
    adata = adata[adata.obs_names.isin(pos.index)].copy()
    pos = pos.loc[adata.obs_names]
    adata.obs[["array_row", "array_col"]] = pos[["array_row", "array_col"]].values
    adata.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float)  # x, y
    sf = json.loads((path / "spatial/scalefactors_json.json").read_text())
    sp_dir = path / "spatial"
    images = {k: imread(sp_dir / f"tissue_{k}_image.png") for k in ("hires", "lowres")
              if (sp_dir / f"tissue_{k}_image.png").exists()}
    adata.uns["spatial"] = {slice_id: {"scalefactors": sf, "images": images}}
    adata.layers["counts"] = adata.X.copy()
    adata.uns["slice_id"] = slice_id
    return annotate_species(adata)

LOADERS = {"visium": load_visium, "stq": load_stq}
def load_slice(cfg, slice_cfg):
    """Load one slice using slice_cfg['loader'] (default 'visium')."""
    path = Path(cfg["paths"]["raw_st"]) / slice_cfg["st"]
    return LOADERS[slice_cfg.get("loader", "visium")](path, slice_cfg["id"])

def coords_to_um(adata: AnnData, pitch_um: float = VISIUM_PITCH_UM) -> None:
    """obsm['spatial'] px → µm, scaled so median nearest-neighbour = pitch_um; px kept in obsm['spatial_px']."""
    from sklearn.neighbors import NearestNeighbors
    px = np.asarray(adata.obsm["spatial"], dtype=float)
    d, _ = NearestNeighbors(n_neighbors=2).fit(px).kneighbors(px)
    um_per_px = pitch_um / float(np.median(d[:, 1]))
    lib = next(iter(adata.uns["spatial"]))
    diam_px = adata.uns["spatial"][lib]["scalefactors"]["spot_diameter_fullres"]
    adata.obsm["spatial_px"] = px
    adata.obsm["spatial"] = px * um_per_px
    adata.uns["um_per_px"] = um_per_px
    adata.uns["um_per_px_from_diameter"] = SPACERANGER_REF_UM / diam_px   # cross-check only


def mito_cutoff(pct: np.ndarray, n_mad: float = 3.0, floor: float = 5.0) -> float:
    """Cutoff = max(median + n_mad·MAD, floor), in percent; NaNs ignored."""
    pct = pct[np.isfinite(pct)]
    if pct.size == 0:
        return floor
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


def preprocess_st(adata, cfg, slice_cfg=None):
    """Convert coords to µm, QC, log-normalize. slice_cfg['qc'] overrides cfg['st']."""
    slice_cfg = slice_cfg or {}
    qc = {**cfg["st"], **slice_cfg.get("qc", {})}          # ← per-slice override
    coords_to_um(adata)
    adata = qc_filter(adata,
              mito_prefix=qc["mito_prefix"],
              min_counts=qc["min_counts"],
              min_genes=qc["min_genes"],
              min_cells=qc.get("min_cells", 1),
              n_mad=qc.get("n_mad", 3.0),
              mito_mode=qc.get("mito_mode", "flag"))
    lognormalize(adata)
    adata.uns["modality"] = "st"
    adata.uns["pixel_size_um"] = cfg["st"].get("spot_diameter_um", 55.0)
    if "model" in slice_cfg:
        adata.uns["model"] = slice_cfg["model"]
    return adata