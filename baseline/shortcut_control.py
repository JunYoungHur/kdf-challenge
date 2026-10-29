"""Control baseline B0: a classifier that only sees non-acoustic metadata.

Features: duration, leading silence, RMS and peak of the PROCESSED audio (exactly the
cues a participant could read off the released files without listening).  It is trained on the
train split and scored on eval.  If the benchmark is free of the classic silence/length
shortcut, this system should sit near 50 % EER.  It is reported on the leaderboard as
evidence, not as a contender.

  python baseline/shortcut_control.py --data_dir kdf_data --out results/scores_shortcut.csv
"""
import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import read_csv, write_csv  # noqa: E402


def feats(rows):
    return np.array([[float(r["dur"]), float(r["proc_lead_sil"]), float(r["rms"]), float(r["peak"])] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="kdf_data")
    ap.add_argument("--out", default="results/scores_shortcut.csv")
    a = ap.parse_args()
    P = read_csv(os.path.join(a.data_dir, "protocol.csv"))
    tr = [r for r in P if r["split"] == "train"]
    ev = [r for r in P if r["split"] == "eval"]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(feats(tr), [r["label"] == "spoof" for r in tr])
    p = clf.predict_proba(feats(ev))[:, 1]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    write_csv(a.out, [{"utt_id": r["utt_id"], "score": f"{s:.6f}"} for r, s in zip(ev, p)],
              ["utt_id", "score"])
    print(f"{len(ev)} scores -> {a.out}")


if __name__ == "__main__":
    main()
