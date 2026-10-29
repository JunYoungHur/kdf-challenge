"""Score a submission against protocol.csv (eval split).

  python evaluation/evaluate.py --protocol kdf_data/protocol.csv --scores results/scores_aasist.csv --name AASIST

Score file: CSV with columns utt_id,score ; higher score = more likely SPOOF.
Every eval utt_id must be present; a missing one is treated as the worst possible score
for its class (bona fide -> +inf, spoof -> -inf), i.e. an error.
Reported: overall EER, EER per generator (that generator's spoofs vs all bona fide),
pooled seen-generator EER and pooled unseen-generator EER.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import read_csv  # noqa: E402
from evaluation.metrics import eer  # noqa: E402


def score(protocol, scores):
    ev = [r for r in protocol if r["split"] == "eval"]
    S = {r["utt_id"]: float(r["score"]) for r in scores}
    missing = 0

    def get(r):
        nonlocal missing
        if r["utt_id"] in S:
            return S[r["utt_id"]]
        missing += 1
        return np.inf if r["label"] == "bonafide" else -np.inf

    y = np.array([r["label"] == "spoof" for r in ev])
    s = np.array([get(r) for r in ev])
    gen = np.array([r["generator"] for r in ev])
    seen = np.array([r["seen"] == "1" for r in ev])
    bona = ~y
    res = {"n_eval": len(ev), "missing": missing, "EER": eer(y, s)[0], "EER_by_generator": {}}
    for g in sorted(set(gen[y])):
        m = bona | (gen == g)
        res["EER_by_generator"][g] = eer(y[m], s[m])[0]
    for name, mask in (("EER_seen", seen), ("EER_unseen", ~seen)):
        m = bona | (y & mask)
        res[name] = eer(y[m], s[m])[0] if (y & mask).any() else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--name", default="system")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = score(read_csv(a.protocol), read_csv(a.scores))
    res["system"] = a.name
    print(json.dumps(res, indent=1))
    f = lambda v: "—" if v is None else f"{100 * v:.2f}"
    gens = sorted(res["EER_by_generator"])
    print("\n| System | EER (all) | EER seen | EER unseen | " + " | ".join(gens) + " |")
    print(f"| {a.name} | {f(res['EER'])} | {f(res['EER_seen'])} | {f(res['EER_unseen'])} | "
          + " | ".join(f(res["EER_by_generator"][g]) for g in gens) + " |")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
