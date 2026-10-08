"""I/O helpers and the AnnData contract."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml
from anndata import AnnData

RAW_LAYER = {"st": "counts", "sm": "raw"}
REQUIRED_UNS = ("modality", "pixel_size_um", "slice_id")


class ContractError(ValueError):
    """Raised when an AnnData violates the project contract."""


def load_config(path: str | Path) -> dict:
    """Load a cohort YAML config."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def _min(x) -> float:
    if sp.issparse(x):
        return float(x.data.min()) if x.nnz else 0.0
    return float(np.min(x))


def validate_adata(adata: AnnData, modality: str) -> None:
    """Check the AnnData contract; raise ContractError listing all violations.

    Contract: raw layer ('counts' for ST, 'raw' for SM), 'lognorm' layer,
    X equal to lognorm and non-negative, obsm['spatial'] (n_obs, 2),
    uns keys modality / pixel_size_um / slice_id.
    """
    if modality not in RAW_LAYER:
        raise ValueError(f"modality must be one of {list(RAW_LAYER)}")
    errors: list[str] = []

    raw = RAW_LAYER[modality]
    for layer in (raw, "lognorm"):
        if layer not in adata.layers:
            errors.append(f"missing layers['{layer}']")

    if adata.n_obs and _min(adata.X) < 0:
        errors.append("X has negative values (scaled data must not be saved)")
    if raw in adata.layers and adata.n_obs and _min(adata.layers[raw]) < 0:
        errors.append(f"layers['{raw}'] has negative values")

    spatial = adata.obsm.get("spatial")
    if spatial is None:
        errors.append("missing obsm['spatial']")
    elif np.asarray(spatial).shape != (adata.n_obs, 2):
        errors.append(f"obsm['spatial'] shape {np.asarray(spatial).shape} != ({adata.n_obs}, 2)")

    for key in REQUIRED_UNS:
        if key not in adata.uns:
            errors.append(f"missing uns['{key}']")
    if adata.uns.get("modality") not in (None, modality):
        errors.append(f"uns['modality']={adata.uns['modality']!r} != {modality!r}")

    if errors:
        raise ContractError(f"{modality} AnnData invalid:\n- " + "\n- ".join(errors))