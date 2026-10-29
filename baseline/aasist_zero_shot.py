"""Baseline B1: AASIST pretrained on ASVspoof 2019 LA (English), applied zero-shot to Korean.

Setup (once):
  git clone https://github.com/clovaai/aasist third_party/aasist
The repository ships config/AASIST.conf and models/weights/AASIST.pth.

  python baseline/aasist_zero_shot.py --data_dir kdf_data --split eval --out results/scores_aasist.csv
Output: utt_id,score  with higher = more likely spoof.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common import load_wav, read_csv, write_csv  # noqa: E402

AASIST_DIR = os.path.join(ROOT, "third_party", "aasist")
N_SAMPLES = 64600  # ~4.04 s at 16 kHz, the length AASIST was trained on


def fix_len(x, n=N_SAMPLES):
    if len(x) >= n:
        return x[:n]
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def load_model(device):
    sys.path.insert(0, AASIST_DIR)
    from models.AASIST import Model
    conf = json.load(open(os.path.join(AASIST_DIR, "config", "AASIST.conf")))
    m = Model(conf["model_config"]).to(device)
    sd = torch.load(os.path.join(AASIST_DIR, "models", "weights", "AASIST.pth"), map_location=device)
    m.load_state_dict(sd)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="kdf_data")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out", default="results/scores_aasist.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    m = load_model(a.device)
    rows = [r for r in read_csv(os.path.join(a.data_dir, "protocol.csv")) if r["split"] == a.split]
    out = []
    with torch.inference_mode():
        for r in tqdm(rows):
            x = fix_len(load_wav(os.path.join(a.data_dir, r["path"])))
            _, logits = m(torch.from_numpy(x)[None].to(a.device))
            bona = float(logits[0, 1])          # AASIST: index 1 = bona fide
            out.append({"utt_id": r["utt_id"], "score": f"{-bona:.6f}"})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    write_csv(a.out, out, ["utt_id", "score"])
    print(f"{len(out)} scores -> {a.out}")


if __name__ == "__main__":
    main()
