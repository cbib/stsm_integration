import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

from stsm_integration.io import validate_adata
from stsm_integration.preprocess.st import lognormalize, mito_cutoff, preprocess_st

rng = np.random.default_rng(0)


def fake_visium(n=200, g=50, n_mt=5):
    counts = rng.poisson(3, size=(n, g)).astype(np.float32)
    counts[:10, :n_mt] += 200          # 10 high-mito spots
    a = AnnData(X=sp.csr_matrix(counts))
    a.var_names = [f"mt-{i}" for i in range(n_mt)] + [f"g{i}" for i in range(g - n_mt)]
    a.layers["counts"] = a.X.copy()
    a.obsm["spatial"] = rng.uniform(0, 2000, size=(n, 2))
    a.uns["spatial"] = {"lib": {"scalefactors": {"spot_diameter_fullres": 110.0}}}
    a.uns["slice_id"] = "A1"
    return a


def test_mito_cutoff_floor():
    assert mito_cutoff(np.zeros(100)) == 5.0


def test_pipeline_contract_and_filtering():
    a = preprocess_st(fake_visium(), {"mito_prefix": "mt-"})
    validate_adata(a, "st")
    assert a.obs["high_mito"].sum() > 0
    assert a.X.min() >= 0


def test_coords_converted_to_um():
    a = preprocess_st(fake_visium(), {})
    np.testing.assert_allclose(a.obsm["spatial"], a.obsm["spatial_px"] * 0.5)  # 55 µm / 110 px


def test_lognorm_matches_scanpy():
    import scanpy as sc
    a = fake_visium()
    lognormalize(a)
    b = AnnData(X=a.layers["counts"].copy())
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    np.testing.assert_allclose(a.layers["lognorm"].toarray(), b.X.toarray(), rtol=1e-5)


def test_counts_untouched():
    a = fake_visium()
    before = a.layers["counts"].toarray().copy()
    a = preprocess_st(a, {})
    assert np.all(np.isin(a.layers["counts"].toarray(), before))

def test_visium_spacing_is_100um():
    # hexagonal grid: 100 µm centre-to-centre, 377.6 px = 65 µm (Space Ranger reference)
    um_per_px = 65.0 / 377.6
    rows, cols = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    x_um = cols * 100 + (rows % 2) * 50
    y_um = rows * 100 * np.sqrt(3) / 2
    a = fake_visium(n=100)
    a.obsm["spatial"] = np.c_[x_um.ravel(), y_um.ravel()] / um_per_px
    a.uns["spatial"]["lib"]["scalefactors"]["spot_diameter_fullres"] = 377.6
    a = preprocess_st(a, {"mito_prefix": "mt-"})
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=2).fit(a.obsm["spatial"]).kneighbors(a.obsm["spatial"])
    assert np.median(d[:, 1]) == pytest.approx(100, abs=2)