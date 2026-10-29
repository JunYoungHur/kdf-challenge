# KDF: Korean Deepfake-voice Detection Challenge

Speech Recognition Challenge 2026 (아주대학교 음성인식) — Team_JunYoungHur_202126881

AI가 합성한 한국어 목소리를 진짜 사람 목소리와 구별하는 챌린지입니다.
참가자는 녹음 하나를 받아 **가짜일수록 높은 점수**를 내는 시스템을 만들고,
학습 때 보지 못한 합성기로 만든 가짜까지 잡아내야 합니다.

```
common.py                   오디오 입출력, 트리밍, 음량 정규화, 프로토콜 CSV (모든 시스템이 공유)
data/download_real.py       Zeroth-Korean(진짜 음성) 다운로드, 화자 분리 분할
data/make_fakes.py          보코더 재합성으로 가짜 음성 생성 (vocos / hifigan / bigvgan)
data/build_protocol.py      진짜·가짜 동일 후처리 → protocol.csv
baseline/shortcut_control.py  B0: 길이·무음·음량만 보는 대조군 (지름길 검증용)
baseline/aasist_zero_shot.py  B1: ASVspoof2019로 학습된 AASIST를 한국어에 그대로 적용
baseline/ssl_probe.py       기말용 참고 시스템: WavLM 특징 + 로지스틱 회귀 (중간 Leaderboard 미포함)
evaluation/evaluate.py      EER (전체 / 합성기별 / seen·unseen)
scripts/run_all.py          전체 재현 (Windows/macOS/Linux 공통, --from 으로 이어서 실행)
scripts/run_all.sh          동일 내용의 bash 버전
results/                    점수 파일
third_party/                외부 저장소 clone 위치 (aasist, BigVGAN, hifi-gan)
```

---

## 1. 배경지식 (이 저장소를 이해하는 데 필요한 전부)

**딥페이크 음성 탐지는 이진 분류입니다.** 입력은 녹음 하나, 출력은 "진짜(bona fide)" 또는 "가짜(spoof)"입니다. 보통 시스템은 확률이나 점수를 내고, 임계값으로 둘을 가릅니다.

**보코더(vocoder)** 는 음성을 "소리의 설계도"에서 실제 파형으로 되돌리는 신경망입니다. 설계도 역할을 하는 것이 **멜 스펙트로그램**(시간축으로 주파수 성분을 그린 그림)입니다. 요즘 음성 합성(TTS)은 글자 → 멜 스펙트로그램 → 보코더 → 파형 순서로 동작하므로, 보코더는 거의 모든 합성 음성의 마지막 단계입니다. 그래서 진짜 음성의 멜 스펙트로그램을 보코더에 넣어 파형을 다시 만들면, **같은 사람이 같은 말을 한 "기계가 만든" 복사본**이 생깁니다. 이 복사본을 가짜 데이터로 쓰는 방식을 **재합성(copy-synthesis)** 이라 하고, 영어권 탐지 데이터셋(WaveFake 등)이 이 방식을 씁니다. 보코더는 언어를 모르기 때문에 한국어에도 그대로 적용됩니다.

**EER(Equal Error Rate)** 은 탐지 분야의 표준 지표입니다. 임계값을 움직이면 "진짜를 가짜로 오판한 비율"과 "가짜를 진짜로 놓친 비율"이 반대로 움직이는데, 두 값이 같아지는 지점의 오류율입니다. 낮을수록 좋고, 임계값 선택과 무관하게 시스템을 비교할 수 있습니다. 50%면 동전 던지기와 같습니다.

**Seen / Unseen 합성기.** 실제 범죄자가 어떤 합성기를 쓸지는 알 수 없습니다. 그래서 학습에 쓴 합성기(seen)로 만든 가짜와, 학습에 전혀 쓰지 않은 합성기(unseen)로 만든 가짜를 평가에서 따로 봅니다. 이 분야에서 성능 차이가 가장 크게 나는 지점이고, 이 챌린지의 핵심 난이도입니다.

**지름길(shortcut) 문제.** 표준 데이터셋 ASVspoof 2019에서 탐지 모델들이 음성 내용이 아니라 **파일 앞뒤 무음 길이**로 진짜와 가짜를 구분하고 있었다는 것이 밝혀졌습니다(Müller et al., 2021, "Speech is Silver, Silence is Golden"). 이 저장소는 그 교훈을 설계에 반영합니다(3절).

---

## 2. 데이터

### 진짜 음성: Zeroth-Korean
- 출처: https://huggingface.co/datasets/kresnik/zeroth_korean (원 프로젝트 https://github.com/goodatlas/zeroth)
- 라이선스: CC BY 4.0 — 출처 표기 시 재배포 가능. 생성된 평가셋을 저장소에 함께 올릴 수 있습니다.
- 내용: 한국어 낭독 음성, 16 kHz. train과 test는 화자가 겹치지 않습니다.
- 사용 방식: Zeroth **test**(약 457문장) 전체 → 본 챌린지 **eval**. Zeroth **train** 화자를 두 그룹으로 나눠 각각 **train**(1,500문장 샘플), **dev**(400문장 샘플). 세 분할의 화자는 서로 겹치지 않습니다.

### 가짜 음성: 보코더 재합성
| 이름 | 모델 | 가중치 입수 | 역할 |
|---|---|---|---|
| vocos | charactr/vocos-mel-24khz | `pip install vocos` 시 자동 | seen |
| hifigan | HiFi-GAN UNIVERSAL_V1 (jik876/hifi-gan) | README의 Google Drive 링크에서 수동 다운로드 | seen (선택) |
| bigvgan | nvidia/bigvgan_v2_22khz_80band_256x | `git clone NVIDIA/BigVGAN` 후 Hugging Face에서 자동 | **unseen** (eval 전용) |

세 모델 모두 공개 가중치이고 영어 위주 데이터로 학습되었습니다. 한국어로 재합성할 때 품질이 다소 떨어질 수 있는데, 탐지 관점에서는 "완벽하지 않은 가짜"가 오히려 쉬운 쪽이므로 결과 해석 시 이 점을 염두에 둡니다. hifigan 가중치를 받지 못하면 `run_all.sh`가 vocos만 seen으로 두고 진행합니다.

### 추가 절차: 보코더에 넣는 방법 (`data/make_fakes.py`)
1. 16 kHz 진짜 음성을 보코더의 샘플링 주파수(22.05 또는 24 kHz)로 리샘플링
2. **그 보코더가 학습 때 쓴 것과 동일한 설정**으로 멜 스펙트로그램 계산. 설정(FFT 크기, hop, 멜 밴드 수, 주파수 범위)이 다르면 소리가 깨집니다. 그래서 스크립트는 각 모델이 제공하는 멜 계산 함수를 그대로 호출합니다(vocos: `feature_extractor`, BigVGAN: `meldataset.get_mel_spectrogram`, HiFi-GAN: `meldataset.mel_spectrogram`).
3. 보코더로 파형 생성
4. 다시 16 kHz로 리샘플링해 저장. 파일명은 원본과 같게 유지해 1:1 대응을 보존

Colab T4 기준 문장 하나에 0.1초 안팎이라, 2,400문장 × 합성기 수만큼 걸립니다.

---

## 3. 평가셋 조립 (`data/build_protocol.py`)

진짜와 가짜를 **같은 규칙**으로 후처리합니다.

- 트리밍: 진짜 파일에서 에너지 기준으로 말소리 구간(시작·끝)을 찾고, **그 구간을 가짜 복사본에도 그대로 적용**합니다. 가짜 파일을 따로 트리밍하지 않습니다. 보코더 출력은 무음 구간에 미세한 잡음을 남기기 때문에 따로 트리밍하면 가짜가 항상 더 길어지고, 길이만으로 구분되는 지름길이 생깁니다. (이 저장소 개발 중 합성 데이터로 테스트하다 실제로 이 지름길이 생겨 대조군 EER이 0%가 나왔고, 위 방식으로 고친 뒤 50%로 돌아왔습니다.)
- 음량: 모든 파일을 RMS −23 dB로 정규화
- 길이: 최대 8초
- 분할: 가짜는 원본 진짜 문장의 분할을 그대로 따릅니다. unseen 합성기는 eval에만 포함됩니다.

결과물 `protocol.csv`의 열: `utt_id, path, split, label, generator, seen, speaker, source_utt, raw_dur, raw_lead_sil, dur, proc_lead_sil, rms, peak`.

---

## 4. Baseline

| 시스템 | 내용 | 학습 |
|---|---|---|
| B0 Shortcut control | 처리된 오디오의 길이·앞 무음·RMS·피크 4개 숫자만 보는 로지스틱 회귀. 평가셋에 지름길이 없다면 EER ≈ 50%여야 함. 순위 경쟁 대상이 아닌 **증거**로 게시 | train 분할 (메타데이터만) |
| B1 AASIST zero-shot | ASVspoof 2019 LA(영어)로 학습된 공개 탐지 모델(NAVER Clova, https://github.com/clovaai/aasist)을 한국어·새 합성기에 그대로 적용 | 없음 |

기말 참고 시스템 `baseline/ssl_probe.py`(WavLM 특징 + 로지스틱 회귀, seen 합성기로 학습)는 중간 Leaderboard에 넣지 않고, 기말에 Proposed Method와 함께 비교합니다.

---

## 5. 평가 (`evaluation/evaluate.py`)

```bash
python evaluation/evaluate.py --protocol kdf_data/protocol.csv --scores results/scores_aasist.csv --name "B1 AASIST"
```
제출 형식: `utt_id,score` CSV, **높을수록 가짜**. eval의 모든 utt_id가 있어야 하며, 빠진 항목은 해당 클래스의 최악 점수로 처리됩니다.

보고 항목: 전체 EER, 합성기별 EER(해당 합성기의 가짜 vs 전체 진짜), seen 합성기 통합 EER, **unseen 합성기 통합 EER(주 지표)**.

규칙
1. 임계값·하이퍼파라미터는 dev에서만 결정합니다.
2. eval 분할의 어떤 파일도 학습·튜닝에 쓰지 않습니다.
3. unseen 합성기(bigvgan)로 만든 음성은 어떤 형태로도 학습에 쓸 수 없습니다. 외부 공개 데이터와 사전학습 모델은 사용 가능하며 내역을 밝힙니다.

---

## 6. 실행

```bash
pip install -r requirements.txt
git clone https://github.com/clovaai/aasist third_party/aasist
git clone https://github.com/NVIDIA/BigVGAN third_party/BigVGAN
# (선택) git clone https://github.com/jik876/hifi-gan third_party/hifi-gan 후
#        UNIVERSAL_V1/generator_v1 과 config.json 을 third_party/hifi-gan/UNIVERSAL_V1/ 에 배치
python scripts/run_all.py            # Windows PowerShell에서도 그대로 동작
# 중간에 멈췄다면: python scripts/run_all.py --from protocol
```
단계별로는 README 상단 파일 설명 순서(`download_real` → `make_fakes` → `build_protocol` → `baseline/*` → `evaluate`)대로 실행하면 됩니다.

---

## 7. 한계 (페이지에도 명시)
- 보코더 재합성은 실제 음성 복제 TTS가 만드는 가짜의 **하위 집합**입니다. 억양·내용까지 새로 만드는 합성은 다루지 않습니다.
- Zeroth-Korean은 낭독 음성이라 전화 통화 같은 자발 발화와 음향 조건이 다릅니다.
- 보코더 가중치는 영어 데이터로 학습된 것입니다.

## References
- Zeroth-Korean (goodatlas), CC BY 4.0.
- Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," ICASSP 2022.
- Müller et al., "Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?" ASVspoof Workshop 2021.
- Frank & Schönherr, "WaveFake: A Data Set to Facilitate Audio Deepfake Detection," NeurIPS Datasets and Benchmarks 2021.
- Lee et al., "BigVGAN: A Universal Neural Vocoder with Large-Scale Training," ICLR 2023.
- Siuzdak, "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders," ICLR 2024.
- Kong et al., "HiFi-GAN," NeurIPS 2020.
- Chen et al., "WavLM," IEEE JSTSP 2022.
- ICASSP 2026 SP Grand Challenge GC-14 WildSpoof (문제 설정 참고).
