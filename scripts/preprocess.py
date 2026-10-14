"""Preprocess every ST slice of a cohort: python scripts/preprocess_st.py configs/vicari_mouse.yaml"""
import argparse
from pathlib import Path

from stsm_integration.io import load_config, validate_adata
from stsm_integration.preprocess.st import load_visium, preprocess_st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw, out = Path(cfg["paths"]["raw_st"]), Path(cfg["paths"]["processed_st"]) / "ST"
    out.mkdir(parents=True, exist_ok=True)

    for s in cfg["slices"]:
        dst = out / f"{s['id']}.h5ad"
        if dst.exists() and not args.overwrite:
            print(f"[skip] {dst} exists")
            continue
        src = raw / s["st"] / "outs"
        adata = preprocess_st(load_visium(str(src), s["id"]), cfg["st"])
        validate_adata(adata, "st")
        adata.write_h5ad(dst)
        q = adata.uns["qc"]
        print(f"[ok] {s['id']}: {q['spots_in']}→{q['spots_out']} spots, "
              f"{q['genes_in']}→{q['genes_out']} genes, mito<{q['mito_cutoff_pct']:.1f}% → {dst}")


if __name__ == "__main__":
    main()