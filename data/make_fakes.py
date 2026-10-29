"""Create spoofed copies of every real utterance by vocoder re-synthesis.

real wav (16 kHz) -> resample to the vocoder's rate -> mel spectrogram -> vocoder -> wav
-> resample back to 16 kHz.  Content and speaker are identical to the real file; only the
waveform is machine-generated.  Output: <data_dir>/fake/<generator>/<split>/<utt_id>.wav

  python data/make_fakes.py --data_dir kdf_data --generator vocos   --splits train dev eval
  python data/make_fakes.py --data_dir kdf_data --generator bigvgan --splits eval
  python data/make_fakes.py --data_dir kdf_data --generator hifigan --splits train dev eval

Generators
  vocos   : charactr/vocos-mel-24khz          pip install vocos          (auto-download)
  bigvgan : nvidia/bigvgan_v2_22khz_80band_256x  git clone NVIDIA/BigVGAN into third_party/ (auto-download weights)
  hifigan : jik876/hifi-gan UNIVERSAL_V1      git clone into third_party/hifi-gan and put
            generator_v1 + config.json in third_party/hifi-gan/UNIVERSAL_V1/ (manual download)
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
from common import SR, load_wav, read_csv, save_wav  # noqa: E402


def get_vocos(device):
    from vocos import Vocos
    m = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()

    def run(x16):
        import librosa
        x = librosa.resample(x16, orig_sr=SR, target_sr=24000)
        with torch.inference_mode():
            y = m(torch.from_numpy(x)[None].to(device)).squeeze().cpu().numpy()
        return librosa.resample(y, orig_sr=24000, target_sr=SR)
    return run


def get_bigvgan(device):
    # Build the model by hand instead of BigVGAN.from_pretrained(): the repo's
    # from_pretrained() is incompatible with recent huggingface_hub versions.
    sys.path.insert(0, os.path.join(ROOT, "third_party", "BigVGAN"))
    from huggingface_hub import hf_hub_download
    from bigvgan import BigVGAN
    from env import AttrDict
    from meldataset import get_mel_spectrogram
    repo = "nvidia/bigvgan_v2_22khz_80band_256x"
    h = AttrDict(json.load(open(hf_hub_download(repo, "config.json"))))
    m = BigVGAN(h, use_cuda_kernel=False)
    ckpt = torch.load(hf_hub_download(repo, "bigvgan_generator.pt"), map_location="cpu")
    m.load_state_dict(ckpt["generator"])
    m.remove_weight_norm()
    m = m.eval().to(device)
    sr = h.sampling_rate

    def run(x16):
        import librosa
        x = librosa.resample(x16, orig_sr=SR, target_sr=sr)
        mel = get_mel_spectrogram(torch.from_numpy(x)[None], h).to(device)
        with torch.inference_mode():
            y = m(mel).squeeze().cpu().numpy()
        return librosa.resample(y, orig_sr=sr, target_sr=SR)
    return run


def get_hifigan(device):
    hp = os.path.join(ROOT, "third_party", "hifi-gan")
    sys.path.insert(0, hp)
    from env import AttrDict
    from meldataset import mel_spectrogram
    from models import Generator
    d = os.path.join(hp, "UNIVERSAL_V1")
    h = AttrDict(json.load(open(os.path.join(d, "config.json"))))
    g = Generator(h).to(device)
    g.load_state_dict(torch.load(os.path.join(d, "generator_v1"), map_location=device)["generator"])
    g.eval(); g.remove_weight_norm()

    def run(x16):
        import librosa
        x = librosa.resample(x16, orig_sr=SR, target_sr=h.sampling_rate)
        mel = mel_spectrogram(torch.from_numpy(x)[None], h.n_fft, h.num_mels, h.sampling_rate,
                              h.hop_size, h.win_size, h.fmin, h.fmax).to(device)
        with torch.inference_mode():
            y = g(mel).squeeze().cpu().numpy()
        return librosa.resample(y, orig_sr=h.sampling_rate, target_sr=SR)
    return run


GENERATORS = {"vocos": get_vocos, "bigvgan": get_bigvgan, "hifigan": get_hifigan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="kdf_data")
    ap.add_argument("--generator", choices=GENERATORS, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "eval"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    run = GENERATORS[a.generator](a.device)
    rows = [r for r in read_csv(os.path.join(a.data_dir, "real_manifest.csv")) if r["split"] in a.splits]
    for r in tqdm(rows, desc=a.generator):
        out = os.path.join(a.data_dir, "fake", a.generator, r["split"], r["utt_id"] + ".wav")
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        y = run(load_wav(os.path.join(a.data_dir, r["path"])))
        save_wav(out, np.clip(y, -1, 1).astype(np.float32))
    print(f"{a.generator}: {len(rows)} files -> {a.data_dir}/fake/{a.generator}")


if __name__ == "__main__":
    main()
