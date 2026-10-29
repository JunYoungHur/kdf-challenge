"""Download real (bona fide) Korean speech from Zeroth-Korean (CC BY 4.0) via Hugging Face.

Zeroth test speakers -> eval ; a speaker-stratified sample of train speakers -> train/dev.
Writes 16 kHz wavs to <data_dir>/real/<split>/ and <data_dir>/real_manifest.csv.

  python data/download_real.py --data_dir kdf_data --n_train 1500 --n_dev 400
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SR, save_wav, write_csv  # noqa: E402


def speaker_of(ex):
    for k in ("speaker_id", "speaker"):
        if k in ex and ex[k] is not None:
            return str(ex[k])
    # zeroth ids look like "104_003_0001" -> speaker 104
    uid = str(ex.get("id") or ex.get("file") or "")
    return os.path.basename(uid).split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="kdf_data")
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_dev", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    import librosa
    from datasets import load_dataset

    ds = load_dataset("kresnik/zeroth_korean")
    rng = np.random.default_rng(a.seed)
    rows = []

    def dump(ex, split, idx):
        spk = speaker_of(ex)
        au = ex["audio"]
        x, sr = au["array"].astype(np.float32), au["sampling_rate"]
        if sr != SR:
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        uid = f"real_{split}_{idx:05d}"
        p = os.path.join(a.data_dir, "real", split, uid + ".wav")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        save_wav(p, x)
        rows.append({"utt_id": uid, "path": os.path.relpath(p, a.data_dir), "split": split,
                     "speaker": spk, "text": ex.get("text", "")})

    # eval = all Zeroth test utterances (speakers disjoint from train by construction)
    for i, ex in enumerate(ds["test"]):
        dump(ex, "eval", i)

    # train/dev: split Zeroth TRAIN speakers into two disjoint groups, then sample
    train = ds["train"]
    meta = train.remove_columns("audio")          # read speaker ids without decoding audio
    spk = np.array([speaker_of(meta[i]) for i in range(len(meta))])
    uniq = rng.permutation(np.unique(spk))
    n_dev_spk = max(2, int(0.2 * len(uniq)))
    dev_spk, tr_spk = set(uniq[:n_dev_spk]), set(uniq[n_dev_spk:])
    tr_idx = rng.permutation([i for i in range(len(train)) if spk[i] in tr_spk])[: a.n_train]
    dv_idx = rng.permutation([i for i in range(len(train)) if spk[i] in dev_spk])[: a.n_dev]
    for j, i in enumerate(tr_idx):
        dump(train[int(i)], "train", j)
    for j, i in enumerate(dv_idx):
        dump(train[int(i)], "dev", j)

    write_csv(os.path.join(a.data_dir, "real_manifest.csv"), rows,
              ["utt_id", "path", "split", "speaker", "text"])
    print(f"real: train={len(tr_idx)} dev={len(dv_idx)} eval={len(ds['test'])} -> {a.data_dir}/real")


if __name__ == "__main__":
    main()
