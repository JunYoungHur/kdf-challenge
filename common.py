"""Shared helpers for the KDF (Korean Deepfake-voice detection) challenge."""
import csv
import numpy as np
import soundfile as sf

SR = 16000
PROTOCOL_FIELDS = ["utt_id", "path", "split", "label", "generator", "seen", "speaker",
                   "source_utt", "raw_dur", "raw_lead_sil", "dur", "proc_lead_sil", "rms", "peak"]


def load_wav(path, target_sr=SR):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
    return x.astype(np.float32)


def save_wav(path, x, sr=SR):
    sf.write(path, x, sr, subtype="PCM_16")


def leading_silence(x, sr=SR, top_db=40.0, frame=320):
    """Seconds of silence before the first energetic frame (used for the shortcut control)."""
    n = len(x) // frame
    if n == 0:
        return 0.0
    e = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    keep = np.where(20 * np.log10(e / (e.max() + 1e-12)) > -top_db)[0]
    return float(keep[0] * frame / sr) if len(keep) else 0.0


def trim_bounds(x, top_db=40.0, frame=320):
    """(start, end) sample indices of the non-silent span, by a frame-energy rule."""
    n = len(x) // frame
    if n == 0:
        return 0, len(x)
    e = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    keep = np.where(20 * np.log10(e / (e.max() + 1e-12)) > -top_db)[0]
    if len(keep) == 0:
        return 0, len(x)
    return int(keep[0] * frame), int((keep[-1] + 1) * frame)


def trim(x, top_db=40.0, frame=320):
    s, e = trim_bounds(x, top_db, frame)
    return x[s:e]


def rms_normalize(x, target_db=-23.0):
    r = np.sqrt(np.mean(x ** 2) + 1e-12)
    y = x * (10 ** (target_db / 20) / r)
    peak = np.abs(y).max()
    return (y * (0.99 / peak) if peak > 0.99 else y).astype(np.float32)


def align_to(x, ref, max_lag_s=0.05, sr=SR):
    """Shift x so it is time-aligned with ref (vocoders can add a few ms of delay).
    The lag is found by cross-correlation on the first 3 s, searched within +-max_lag_s."""
    n = min(len(x), len(ref), 3 * sr)
    if n < 2048:
        return x, 0
    a, b = ref[:n] - ref[:n].mean(), x[:n] - x[:n].mean()
    L = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.irfft(np.fft.rfft(b, L) * np.conj(np.fft.rfft(a, L)), L)
    m = int(max_lag_s * sr)
    lags = np.r_[np.arange(0, m + 1), np.arange(-m, 0)]
    vals = np.r_[c[: m + 1], c[-m:]]
    lag = int(lags[np.argmax(vals)])          # x is delayed by `lag` samples
    if lag > 0:
        x = x[lag:]
    elif lag < 0:
        x = np.concatenate([np.zeros(-lag, np.float32), x])
    return x.astype(np.float32), lag


def standardize(x, bounds, inner_s=0.05, fade_s=0.02, max_sec=8.0):
    """Identical post-processing for a real file and all of its (aligned) fakes.

    `bounds` are the speech boundaries measured on the REAL source.  They sit exactly where the
    real file crosses the energy threshold, so cutting there would make the real file's first
    frame "just above threshold" by construction while a fake's equivalent frame lands above
    or below it by chance -- an onset-energy cue created by the cut itself.  The span is
    therefore pulled inward by `inner_s` on both sides so both edges fall inside speech, then
    the same sample range is cut from every copy (no silence is kept: recording noise and
    vocoder noise differ), and the same fade-in/out and RMS normalisation are applied."""
    s, e = bounds
    m = int(inner_s * SR)
    if e - s > 4 * m:
        s, e = s + m, e - m
    y = np.zeros(e, np.float32)
    k = min(e, len(x))
    y[:k] = x[:k]
    y = y[s:e]
    f = int(fade_s * SR)
    if len(y) > 2 * f:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        y[:f] *= ramp
        y[-f:] *= ramp[::-1]
    return rms_normalize(y)[: int(max_sec * SR)]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
