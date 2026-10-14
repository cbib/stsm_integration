"""Preprocess every ST slice of a cohort: python scripts/preprocess.py configs/<cohort>.yaml [--overwrite]"""
import argparse
from pathlib import Path

from stsm_integration.io import load_config, validate_adata
from stsm_integration.preprocess.st import load_slice, preprocess_st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg["paths"]["processed_st"])
    out.mkdir(parents=True, exist_ok=True)

    for s in cfg["slices"]:
        dst = out / f"{s['id']}.h5ad"
        if dst.exists() and not args.overwrite:
            print(f"[skip] {dst} exists")
            continue
        adata = preprocess_st(load_slice(cfg, s), cfg, slice_cfg=s)
        validate_adata(adata, "st")
        adata.write_h5ad(dst)
        q = adata.uns["qc"]
        print(f"[ok] {s['id']} ({adata.uns['genome']}): {q['spots_in']}→{q['spots_out']} spots, "
              f"floors {q['min_counts']}/{q['min_genes']}, {q['genes_in']}→{q['genes_out']} genes, "
              f"{adata.uns['um_per_px']:.4f} µm/px → {dst}")


if __name__ == "__main__":
    main()