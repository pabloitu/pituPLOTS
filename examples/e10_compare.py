"""
E10. Model comparison
=====================

Two ESHM20 branches on a shared scale and their percent difference on a
symmetric one. No reference figure.
"""
import os
from pathlib import Path

from pituplots import compare_grids, theme
from pituplots.io import read_csep_grid

data = Path(os.environ["PITUPLOTS_DATA"])
out = Path(os.environ.get("PITUPLOTS_OUT", Path(__file__).parent / "output"))
out.mkdir(parents=True, exist_ok=True)

d = data / "testing_eshm20/eshm_20"
a = read_csep_grid(d / "asm_ver12e_winGT_fs017_lo_abgrs_maxmag_mid.csv")
b = read_csep_grid(d / "asm_ver12e_winGT_fs017_hi_abgrs_maxmag_mid.csv")
with theme("report"):
    fig, axs = compare_grids(a, b, log=True, kind="pct", titles=("low b", "high b", "low vs high"), figsize=(15, 5),
                             clabel=r"$\log_{10}$ rate", dlabel="% difference")
    for s in ("png", "pdf"):
        fig.savefig(out / f"e10_compare.{s}", dpi=200, bbox_inches="tight")
