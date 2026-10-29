"""Per-feature shortcut check: EER of each non-acoustic statistic alone, per generator.
A value far below 50 % means that statistic alone separates real from that generator's fakes.

  python evaluation/check_shortcut.py --protocol kdf_data/protocol.csv
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import read_csv  # noqa: E402
from evaluation.metrics import eer  # noqa: E402

FEATS = ["dur", "proc_lead_sil", "rms", "peak"]

ap = argparse.ArgumentParser()
ap.add_argument("--protocol", default="kdf_data/protocol.csv")
a = ap.parse_args()
ev = [r for r in read_csv(a.protocol) if r["split"] == "eval"]
gens = sorted({r["generator"] for r in ev} - {"real"})
print("feature        " + "".join(f"{g:>10s}" for g in gens))
for f in FEATS:
    out = []
    for g in gens:
        rows = [r for r in ev if r["generator"] in ("real", g)]
        y = np.array([r["label"] == "spoof" for r in rows])
        s = np.array([float(r[f]) for r in rows])
        out.append(100 * min(eer(y, s)[0], eer(y, -s)[0]))
    print(f"{f:14s} " + "".join(f"{v:9.2f}%" for v in out))
