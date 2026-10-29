"""Assemble the benchmark: identical post-processing for real and fake, then protocol.csv.

Every real file is trimmed by an energy rule.  Each vocoded copy is first time-aligned to its
real source (cross-correlation), then the SAME sample range (speech only, no silence margin) is
cut from real and fake alike; an identical 20 ms fade-in/out and RMS normalisation to
-23 dB follow, and length is capped at 8 s.  Duration, silence and loudness therefore carry no label information.  Simple
statistics of the processed audio are recorded in protocol.csv so the shortcut control
baseline can verify this.

Seen generators appear in train/dev/eval; unseen generators appear in eval only.

  python data/build_protocol.py --data_dir kdf_data --seen vocos hifigan --unseen bigvgan
"""
import argparse
import os
import sys

import numpy as np

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PROTOCOL_FIELDS, SR, align_to, leading_silence, load_wav, read_csv, save_wav, standardize, trim_bounds, write_csv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="kdf_data")
    ap.add_argument("--seen", nargs="+", default=["vocos", "hifigan"])
    ap.add_argument("--unseen", nargs="+", default=["bigvgan"])
    a = ap.parse_args()

    real = read_csv(os.path.join(a.data_dir, "real_manifest.csv"))
    gens = [(g, "1") for g in a.seen] + [(g, "0") for g in a.unseen]
    rows = []

    bounds, real_path, lags = {}, {r["utt_id"]: r["path"] for r in real}, []

    def process(src_path, out_path, meta):
        x = load_wav(os.path.join(a.data_dir, src_path))
        raw_dur, raw_lead = len(x) / SR, leading_silence(x)
        if meta["generator"] == "real":
            bounds[meta["utt_id"]] = trim_bounds(x)
        else:  # undo any vocoder delay before cutting the real-defined span
            ref = load_wav(os.path.join(a.data_dir, real_path[meta["source_utt"]]))
            x, lag = align_to(x, ref)
            lags.append((meta["generator"], lag))
        y = standardize(x, bounds[meta["source_utt"]])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_wav(out_path, y)
        rows.append({**meta, "path": os.path.relpath(out_path, a.data_dir),
                     "raw_dur": f"{raw_dur:.3f}", "raw_lead_sil": f"{raw_lead:.3f}",
                     "dur": f"{len(y) / SR:.3f}", "proc_lead_sil": f"{leading_silence(y):.3f}",
                     "rms": f"{float(np.sqrt(np.mean(y ** 2))):.5f}", "peak": f"{float(np.abs(y).max()):.5f}"})

    for r in tqdm(real, desc="real"):
        out = os.path.join(a.data_dir, "proc", "real", r["split"], r["utt_id"] + ".wav")
        process(r["path"], out, {"utt_id": r["utt_id"], "split": r["split"], "label": "bonafide",
                                 "generator": "real", "seen": "1", "speaker": r["speaker"],
                                 "source_utt": r["utt_id"]})
    for g, seen in gens:
        for r in tqdm(real, desc=g):
            if seen == "0" and r["split"] != "eval":
                continue
            src = os.path.join("fake", g, r["split"], r["utt_id"] + ".wav")
            if not os.path.exists(os.path.join(a.data_dir, src)):
                continue
            uid = f"{g}_{r['utt_id']}"
            out = os.path.join(a.data_dir, "proc", g, r["split"], uid + ".wav")
            process(src, out, {"utt_id": uid, "split": r["split"], "label": "spoof", "generator": g,
                               "seen": seen, "speaker": r["speaker"], "source_utt": r["utt_id"]})

    write_csv(os.path.join(a.data_dir, "protocol.csv"), rows, PROTOCOL_FIELDS)
    for g in sorted({g for g, _ in lags}):
        v = np.array([l for gg, l in lags if gg == g])
        print(f"alignment {g:8s} median lag {np.median(v):+.0f} samples, |lag|>0 in {np.mean(v != 0):.0%} of files")
    from collections import Counter
    c = Counter((r["split"], r["generator"]) for r in rows)
    for k in sorted(c):
        print(f"{k[0]:5s} {k[1]:8s} {c[k]}")
    print(f"protocol -> {a.data_dir}/protocol.csv ({len(rows)} files)")


if __name__ == "__main__":
    main()
