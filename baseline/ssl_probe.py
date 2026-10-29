"""Reference system for the FINAL phase (not part of the mid-term leaderboard):
frozen WavLM features + logistic regression trained on the seen generators.

  python baseline/ssl_probe.py --data_dir kdf_data --out results/scores_wavlm_probe.csv
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_wav, read_csv, write_csv  # noqa: E402


def embed_all(rows, data_dir, model_name, device, layer):
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    m = WavLMModel.from_pretrained(model_name).to(device).eval()
    X = []
    with torch.inference_mode():
        for r in tqdm(rows):
            x = load_wav(os.path.join(data_dir, r["path"]))[: 16000 * 6]
            inp = fe(x, sampling_rate=16000, return_tensors="pt").to(device)
            h = m(**inp, output_hidden_states=True).hidden_states[layer]
            X.append(h.mean(1).squeeze().cpu().numpy())
    return np.stack(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="kdf_data")
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--out", default="results/scores_wavlm_probe.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    P = read_csv(os.path.join(a.data_dir, "protocol.csv"))
    tr = [r for r in P if r["split"] == "train"]
    ev = [r for r in P if r["split"] == "eval"]
    Xtr = embed_all(tr, a.data_dir, a.model, a.device, a.layer)
    Xev = embed_all(ev, a.data_dir, a.model, a.device, a.layer)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    clf.fit(Xtr, [r["label"] == "spoof" for r in tr])
    p = clf.predict_proba(Xev)[:, 1]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    write_csv(a.out, [{"utt_id": r["utt_id"], "score": f"{s:.6f}"} for r, s in zip(ev, p)],
              ["utt_id", "score"])
    print(f"{len(ev)} scores -> {a.out}")


if __name__ == "__main__":
    main()
