#!/usr/bin/env bash
# End-to-end reproduction (Colab T4: roughly 1-2 h, dominated by vocoder synthesis).
set -e
DATA=${DATA:-kdf_data}

# 0) third-party code
[ -d third_party/aasist ]  || git clone https://github.com/clovaai/aasist third_party/aasist
[ -d third_party/BigVGAN ] || git clone https://github.com/NVIDIA/BigVGAN third_party/BigVGAN
# hifigan: git clone https://github.com/jik876/hifi-gan third_party/hifi-gan
#          + download UNIVERSAL_V1 (generator_v1, config.json) from the repo's README link
#          into third_party/hifi-gan/UNIVERSAL_V1/

# 1) real speech
python data/download_real.py --data_dir $DATA

# 2) fakes: seen generators on all splits, unseen generator on eval only
python data/make_fakes.py --data_dir $DATA --generator vocos   --splits train dev eval
if [ -f third_party/hifi-gan/UNIVERSAL_V1/generator_v1 ]; then
  python data/make_fakes.py --data_dir $DATA --generator hifigan --splits train dev eval
  SEEN="vocos hifigan"
else
  echo "hifigan weights not found: using vocos as the only seen generator"; SEEN="vocos"
fi
python data/make_fakes.py --data_dir $DATA --generator bigvgan --splits eval

# 3) identical post-processing + protocol
python data/build_protocol.py --data_dir $DATA --seen $SEEN --unseen bigvgan

# 4) baselines
python baseline/shortcut_control.py --data_dir $DATA --out results/scores_shortcut.csv
python baseline/aasist_zero_shot.py --data_dir $DATA --out results/scores_aasist.csv

# 5) scores
python evaluation/evaluate.py --protocol $DATA/protocol.csv --scores results/scores_shortcut.csv --name "B0 shortcut control" --out results/score_shortcut.json
python evaluation/evaluate.py --protocol $DATA/protocol.csv --scores results/scores_aasist.csv   --name "B1 AASIST zero-shot"  --out results/score_aasist.json
