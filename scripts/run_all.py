"""Cross-platform end-to-end run (Windows PowerShell / cmd, macOS, Linux). Same steps as run_all.sh.

  python scripts/run_all.py                 # everything
  python scripts/run_all.py --from protocol # resume from a later step

Steps: clone -> real -> fakes -> protocol -> baselines -> eval
Finished vocoder outputs are skipped automatically, so re-running after an error is cheap.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
STEPS = ["clone", "real", "fakes", "protocol", "baselines", "eval"]


def run(*args):
    print("\n>>", " ".join(args), flush=True)
    subprocess.run(list(args), cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="kdf_data")
    ap.add_argument("--from", dest="start", choices=STEPS, default="clone")
    a = ap.parse_args()
    D = a.data_dir
    todo = STEPS[STEPS.index(a.start):]
    tp = os.path.join(ROOT, "third_party")
    hifi_w = os.path.join(tp, "hifi-gan", "UNIVERSAL_V1", "generator_v1")
    seen = ["vocos", "hifigan"] if os.path.exists(hifi_w) else ["vocos"]

    if "clone" in todo:
        for name, url in [("aasist", "https://github.com/clovaai/aasist"),
                          ("BigVGAN", "https://github.com/NVIDIA/BigVGAN")]:
            if not os.path.isdir(os.path.join(tp, name)):
                run("git", "clone", url, os.path.join("third_party", name))
        if not os.path.exists(hifi_w):
            print("\n[info] HiFi-GAN weights not found -> vocos is the only seen generator.")
    if "real" in todo:
        run(PY, "data/download_real.py", "--data_dir", D)
    if "fakes" in todo:
        for g in seen:
            run(PY, "data/make_fakes.py", "--data_dir", D, "--generator", g, "--splits", "train", "dev", "eval")
        run(PY, "data/make_fakes.py", "--data_dir", D, "--generator", "bigvgan", "--splits", "eval")
    if "protocol" in todo:
        run(PY, "data/build_protocol.py", "--data_dir", D, "--seen", *seen, "--unseen", "bigvgan")
    if "baselines" in todo:
        run(PY, "baseline/shortcut_control.py", "--data_dir", D, "--out", "results/scores_shortcut.csv")
        run(PY, "baseline/aasist_zero_shot.py", "--data_dir", D, "--out", "results/scores_aasist.csv")
    if "eval" in todo:
        P = os.path.join(D, "protocol.csv")
        run(PY, "evaluation/evaluate.py", "--protocol", P, "--scores", "results/scores_shortcut.csv",
            "--name", "B0 shortcut control", "--out", "results/score_shortcut.json")
        run(PY, "evaluation/evaluate.py", "--protocol", P, "--scores", "results/scores_aasist.csv",
            "--name", "B1 AASIST zero-shot", "--out", "results/score_aasist.json")
        run(PY, "evaluation/check_shortcut.py", "--protocol", P)


if __name__ == "__main__":
    main()
