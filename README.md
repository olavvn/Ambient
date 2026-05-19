# AmbientFlow

MIDI 또는 오디오 입력에서 테마를 추출하거나, **미리 준비한 스타일 MIDI 폴더를 테마로 사용**하여 앰비언트 음악을 끊임없이 실시간 생성하는 Python 시스템.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [디렉토리 구조](#3-디렉토리-구조)
4. [구현된 컴포넌트](#4-구현된-컴포넌트)
5. [환경 설정](#5-환경-설정)
6. [실행 방법](#6-실행-방법)
7. [훈련 방법](#7-훈련-방법)
8. [테스트](#8-테스트)
9. [설정 파일 가이드](#9-설정-파일-가이드)
10. [트러블슈팅](#10-트러블슈팅)
11. [구현 현황 및 다음 단계](#11-구현-현황-및-다음-단계)

---

## 1. 프로젝트 개요

### 핵심 아이디어

| 출처 | 아이디어 | 본 프로젝트 적용 |
|------|---------|----------------|
| ThemeTransformer | 테마 기반 컨디셔닝 (대조 학습 + Gated Parallel Attention) | 입력에서 테마를 추출하고, 해당 테마를 반복·변형하며 생성 |
| Magenta RealTime | 실시간 연속 스트리밍 생성 | 청크 단위 롤링 생성, 새 입력에 즉각 반응 |
| 본 프로젝트 | 앰비언트 장르 특화 | 느린 템포, 텍스처, 롱레인지 반복, 부드러운 전환 |

### 전체 흐름

```
사용자 입력 (MIDI 파일 / 오디오 파일 / 실시간 MIDI)
        │
        ▼
   Input Handler
   (MIDIFileHandler / AudioHandler)
        │  pretty_midi 객체
        ▼
  Theme Extractor
  ┌─ 대조 학습 인코더 (SegmentEncoder)
  ├─ 클러스터링 (DBSCAN / K-Means)
  └─ 테마 선택 → 테마 토큰 시퀀스
        │
        ▼
   Theme Buffer (스레드 안전)
        │
        ▼
 Chunk Generator (생성 스레드)
  └─ AmbientFlowModel.generate_chunk()
     ├─ ThemeEncoder: 테마 토큰 → 컨텍스트 벡터
     └─ AmbientMusicDecoder + GPA: 자기회귀 생성
        │  토큰 청크 (128토큰)
        ▼
   Token Queue (asyncio)
        │
        ▼
  Realtime Renderer (재생 스레드)
  └─ 토큰 → MIDI → FluidSynth → 오디오 출력
```

---

## 2. 시스템 아키텍처

### 모델 구조

```
ThemeEncoder
  theme_tokens (B, T)
       │ Embedding + TransformerEncoder
       ▼
  theme_ctx (B, T, D)
       │
       ├──────────────────────────────────┐
       │                                  │
AmbientMusicDecoder                       │
  target_tokens (B, L)                    │
       │ Embedding                        │
       ▼                                  │
  N × DecoderLayer                        │
    ├─ LayerNorm                          │
    ├─ GPAModule ◄──────────── theme_ctx ─┘
    │   ├─ Causal Self-Attention
    │   ├─ Theme Cross-Attention
    │   └─ output = self_out + sigmoid(gate) * theme_out
    ├─ LayerNorm
    └─ FeedForward (GELU)
       │
  logits (B, L, vocab_size)
```

### Gated Parallel Attention (GPA)

ThemeTransformer 논문의 핵심 기여. 매 디코더 레이어에서 일반 자기 어텐션과 테마 크로스 어텐션을 학습 가능한 게이트로 혼합한다.

```
output = self_attn(x) + sigmoid(gate(x)) * theme_cross_attn(x, theme_ctx)
```

게이트가 각 위치, 각 차원별로 테마 참조 강도를 동적으로 조절하므로 모델이 필요할 때만 테마를 참조한다.

### 스트리밍 구조

```
[입력 스레드]         [생성 스레드]           [재생 스레드]
     │                    │                       │
 load_input()             │                       │
 ThemeExtractor           │                       │
     │ theme_tokens        │                       │
     ▼                    │                       │
 ThemeBuffer.update() ──► ChunkGenerator          │
 (threading.Lock)         │ generate_chunk()      │
                          │ 128토큰               │
                          ▼                       │
                      TokenQueue ──────────► RealtimeRenderer
                      (asyncio.Queue)        tokens_to_midi()
                                             FluidSynth
                                             crossfade_audio()
```

### 스타일 폴더 모드 (StyleLibrary)

```
styles/ 폴더 (사용자 준비)
  ├── ballad.mid
  ├── cinematic.mid      ──► StyleLibrary.get_blended_theme()
  └── dark_ambient.mid        각 파일에서 균등 세그먼트 추출 → 연결
                                     │ theme_tokens (혼합 스타일)
                                     ▼
입력 멜로디 (선택)          ThemeBuffer.update()
  melody.mid ──────────►       │
  tokenize()            ChunkGenerator.reset_context(melody_seed)
  │ context seed               │ generate_chunk(theme=스타일, context=멜로디)
  ▼                            ▼
ChunkGenerator.context   TokenQueue → RealtimeRenderer
```

**테마(스타일)** 와 **입력 멜로디** 가 완전히 분리된다:
- `theme_tokens` = styles/ 폴더에서 블렌딩한 스타일
- `context` = 입력 멜로디를 시드로 한 생성 문맥

---

## 3. 디렉토리 구조

```
ambientflow/
├── docs/                          # 설계 문서
│   ├── CLAUDE.md                  # Claude Code 작업 지시서
│   ├── 00_PROJECT_OVERVIEW.md
│   ├── 01_ENVIRONMENT_SETUP.md
│   ├── 02_MUSIC_REPRESENTATION.md
│   ├── 03_THEME_EXTRACTOR.md
│   ├── 04_MODEL_ARCHITECTURE.md
│   ├── 05_STREAMING_ENGINE.md
│   ├── 06_INPUT_HANDLER.md
│   ├── 07_TRAINING_PIPELINE.md
│   ├── 08_INTERFACE_AND_MAIN.md
│   └── 09_TESTING.md
│
├── src/
│   ├── input/
│   │   ├── midi_handler.py        # MIDI 파일 로드 + 실시간 MIDI 수집
│   │   ├── audio_handler.py       # 오디오 → MIDI 변환 (basic-pitch)
│   │   └── normalizer.py          # 입력 경로 통합 팩토리
│   │
│   ├── theme/
│   │   ├── tokenizer.py           # REMIAmbientTokenizer (MIDI ↔ 토큰)
│   │   ├── contrastive.py         # SegmentEncoder + NTXentLoss
│   │   ├── clustering.py          # cluster_segments, select_theme
│   │   ├── extractor.py           # ThemeExtractor (입력에서 자동 추출)
│   │   └── style_library.py       # StyleLibrary (폴더 MIDI → 블렌딩 테마)
│   │
│   ├── model/
│   │   ├── config.py              # ModelConfig (하이퍼파라미터)
│   │   ├── gated_attention.py     # GPAModule (핵심 모듈)
│   │   ├── decoder.py             # DecoderLayer, AmbientMusicDecoder
│   │   └── transformer.py         # ThemeEncoder, AmbientFlowModel
│   │
│   ├── streaming/
│   │   ├── buffer.py              # ThemeBuffer, TokenQueue
│   │   ├── crossfade.py           # crossfade_audio, TokenLevelCrossfade
│   │   ├── chunk_generator.py     # ChunkGenerator (생성 스레드)
│   │   └── engine.py              # AmbientFlowEngine (전체 오케스트레이터)
│   │
│   ├── output/
│   │   ├── renderer.py            # RealtimeRenderer (FluidSynth)
│   │   ├── midi_writer.py         # MIDI 파일 저장
│   │   └── postprocess.py         # 다이나믹스 스무딩, 로우패스 필터
│   │
│   └── utils/
│       ├── config_loader.py       # YAML 설정 로더
│       └── logger.py              # 공통 로거
│
├── training/
│   ├── dataset.py                 # ContrastiveDataset, ThemeGenerationDataset
│   ├── losses.py                  # ThemeAwareCrossEntropy
│   ├── train_contrastive.py       # Phase 1: 대조 학습 훈련
│   └── train_generator.py         # Phase 3: 생성 모델 훈련
│
├── configs/
│   ├── model_config.yaml          # 모델 하이퍼파라미터 (Small 설정)
│   ├── training_config.yaml       # 훈련 설정 (배치, LR, 에폭)
│   └── streaming_config.yaml      # 스트리밍/오디오/FluidSynth 설정
│
├── tests/
│   ├── test_tokenizer.py          # 토크나이저 왕복 변환 (8개)
│   ├── test_theme_extractor.py    # 테마 추출 + 대조 학습 (4개)
│   ├── test_model.py              # 모델 포워드/생성 (5개)
│   ├── test_streaming.py          # 버퍼/큐/크로스페이드 (6개)
│   ├── test_style_library.py      # StyleLibrary 블렌딩/리로드 (10개)
│   └── test_integration.py        # 전체 파이프라인 (3개)
│
├── scripts/
│   ├── check_environment.py       # 환경 검증
│   └── preprocess_data.py         # MIDI 전처리 → PKL 저장
│
├── styles/                        # 스타일 MIDI 폴더 (StyleLibrary 모드용)
│   ├── ballad.mid                 # 예시 — 원하는 스타일 파일 추가
│   └── README.md                  # 스타일 파일 추가 방법 안내
│
├── data/
│   ├── raw_midi/                  # 원본 MIDI 파일 (직접 준비)
│   └── soundfonts/                # FluidSynth 사운드폰트 (직접 준비)
│
├── data_pkl/                      # 전처리된 토큰 데이터 (생성됨)
├── checkpoints/                   # 모델 체크포인트 (훈련 후 생성됨)
│   ├── contrastive/
│   └── generation/
│
├── main.py                        # CLI 진입점
├── pyproject.toml
└── requirements.txt
```

---

## 4. 구현된 컴포넌트

### 4-1. 토크나이저 (`src/theme/tokenizer.py`)

REMI+ 방식을 앰비언트에 맞게 변형한 **REMIAmbientTokenizer**.

**어휘 구성 (총 ~238 토큰):**

| 종류 | 토큰 예시 | 개수 |
|------|----------|------|
| 특수 | `[PAD]` `[BOS]` `[EOS]` `[THEME_REF]` | 8 |
| 마디/박자/위치 | `BAR` `POS_0`~`POS_31` | 37 |
| 템포 | `TEMPO_40`~`TEMPO_120` (5BPM 단계) | 17 |
| 음고 | `PITCH_0`~`PITCH_127` | 128 |
| 음길이 | `DUR_1`~`DUR_32` `DUR_48` `DUR_64` | 34 |
| 세기 | `VEL_8`~`VEL_106` (8단계) | 8 |
| 서스테인 | `SUSTAIN_ON` `SUSTAIN_OFF` | 2 |
| 악기 | `INST_PIANO` `INST_PAD` `INST_BASS` `INST_MELODY` | 4 |

**주요 메서드:**

```python
tokenizer = REMIAmbientTokenizer()

# MIDI → 토큰
tokens = tokenizer.midi_to_tokens(midi, theme_spans=[(0.0, 2.0)])

# 토큰 → MIDI
midi = tokenizer.tokens_to_midi(tokens, bpm=60.0)

# 문자열 ↔ ID 변환
ids = tokenizer.encode(["BAR", "PITCH_60", "DUR_4"])
strs = tokenizer.decode([8, 50, 200])
```

---

### 4-2. 테마 소스 (`src/theme/`)

테마를 공급하는 방법이 두 가지다.

#### 방법 A — StyleLibrary (스타일 폴더, 권장)

원하는 스타일의 MIDI 파일들을 `styles/` 폴더에 넣어두면, 그것들을 혼합해서 테마로 사용한다. 입력 멜로디와 테마가 완전히 분리된다.

```python
from src.theme.style_library import StyleLibrary

lib = StyleLibrary("styles/", max_theme_len=128)
print(lib.style_names)   # ['ballad.mid', 'cinematic.mid', 'dark.mid']

# 전체 블렌딩
theme_tokens = lib.get_blended_theme()

# 일부만 선택
theme_tokens = lib.get_blended_theme(names=["ballad.mid", "cinematic.mid"])

# 텐서로 바로 받기
theme_tensor = lib.get_blended_theme_tensor(device="cuda")  # (1, T)

# 핫 리로드 (파일 추가/삭제 반영)
lib.reload()
```

**블렌딩 방식:** 각 파일에서 `max_theme_len / 파일 수` 길이만큼 세그먼트를 추출해 이어붙인다.

| strategy | 추출 위치 | 특징 |
|---------|----------|------|
| `front` (기본) | 파일 앞부분 | 도입부 스타일 반영 |
| `center` | 파일 중간 | 전개 스타일 반영 |
| `random` | 매번 랜덤 | 다양성 |

#### 방법 B — ThemeExtractor (입력에서 자동 추출)

별도 스타일 파일 없이 입력 MIDI 자체에서 반복 패턴을 찾아 테마로 사용한다.

```python
extractor = ThemeExtractor(
    encoder_checkpoint="checkpoints/contrastive/best_encoder.pt",
    device="cpu",
)
theme_tokens, theme_spans = extractor.extract_from_midi(midi)
```

**대조 학습 (`SegmentEncoder` + `NTXentLoss`):**
- 같은 곡의 두 구간 → positive pair → 임베딩 공간에서 가깝게
- 다른 곡의 구간 → negative pair → 멀게
- NT-Xent loss, temperature=0.07

---

### 4-3. 생성 모델 (`src/model/`)

**ModelConfig 주요 파라미터 (Small 설정):**

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `d_model` | 256 | 모델 차원 |
| `max_seq_len` | 512 | 최대 컨텍스트 길이 |
| `decoder_layers` | 4 | 디코더 레이어 수 |
| `gpa_gate_bias` | -1.0 | 초기 게이트 (낮을수록 테마 영향 약함) |
| `temperature` | 0.95 | 샘플링 온도 |
| `top_p` | 0.92 | Nucleus sampling |

**훈련된 모델 로드 및 생성:**

```python
from src.model.config import ModelConfig
from src.model.transformer import AmbientFlowModel
import torch, yaml

cfg = ModelConfig(**yaml.safe_load(open("configs/model_config.yaml")))
model = AmbientFlowModel(cfg)
ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

theme_tokens = torch.tensor([theme_ids], dtype=torch.long)  # (1, T)
context = torch.tensor([[1]], dtype=torch.long)              # BOS

new_tokens = model.generate_chunk(
    theme_tokens=theme_tokens,
    context=context,
    n_new_tokens=128,
)
```

---

### 4-4. 스트리밍 엔진 (`src/streaming/`)

**ThemeBuffer** — 스레드 안전한 테마 교체:
```python
buf = ThemeBuffer()
buf.update(theme_tensor)         # 입력 스레드
theme = buf.get()                # 생성 스레드
is_new = buf.consume_new_flag()  # 새 테마 여부 확인 후 플래그 초기화
```

**ChunkGenerator** — 무한 롤링 생성:
- 생성 스레드에서 독립 실행
- ThemeBuffer에서 주기적으로 테마 확인
- 새 테마 감지 시 컨텍스트는 유지하고 부드럽게 전환
- 생성된 청크를 TokenQueue에 삽입

**crossfade_audio** — Equal-power 크로스페이드:
```python
faded = crossfade_audio(audio_out, audio_in, fade_samples=4410)
# fade_samples=4410 → 0.1초 @ 44100Hz
```

---

### 4-5. 입력 핸들러 (`src/input/`)

```python
from src.input.normalizer import load_input

# MIDI 파일
midi = load_input(path="song.mid")

# 오디오 파일 (basic-pitch로 기본음 추출)
midi = load_input(path="song.wav")

# 실시간 MIDI 포트 (4초 수집)
midi = load_input(realtime_port="USB MIDI", realtime_sec=4.0)
```

모든 경로는 `pretty_midi.PrettyMIDI` 객체로 정규화된다:
- 노트 퀀타이즈: 32분음표(1/8박자) 그리드
- 템포 클리핑: 40~120 BPM 범위 유지

---

## 5. 환경 설정

### 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| OS | Windows 10 / Ubuntu 20.04 | Ubuntu 22.04 |
| Python | 3.9+ | 3.10 |
| GPU | NVIDIA 8GB (훈련) | 24GB |
| RAM | 16GB | 32GB |

### 패키지 설치

```bash
# 핵심 패키지 (현재 검증 완료)
pip install torch pretty_midi scikit-learn einops pytest pytest-asyncio

# 전체 설치 (requirements.txt 기준)
pip install -r requirements.txt

# FluidSynth (Linux/macOS)
sudo apt-get install -y fluidsynth fluid-soundfont-gm
pip install pyfluidsynth

# 오디오 → MIDI 변환
pip install basic-pitch

# MIDI 실시간 입력
pip install python-rtmidi
```

### 개발 모드 설치

```bash
pip install -e .
```

### 환경 검증

```bash
python scripts/check_environment.py
```

출력 예시:
```
=== Environment Check ===
[OK] PyTorch CUDA: 12.1
[OK] pretty_midi: 0.2.11
[OK] librosa: 0.10.1
[OK] basic-pitch: ok
[FAIL] pyfluidsynth: No module named 'fluidsynth'
[OK] scikit-learn: 1.4.0
[OK] einops: 0.7.0

1 check(s) failed. Please review setup.
```

---

## 6. 실행 방법

> **사전 조건:** 모델 훈련이 완료되어 `checkpoints/best_model.pt`가 존재해야 한다. 훈련 방법은 [7절](#7-훈련-방법) 참조.

### 6-1. MIDI 파일 입력 → 실시간 재생

```bash
python main.py --input examples/theme.mid --device cpu
```

### 6-2. 오디오 파일 입력 → 파일 저장

```bash
python main.py \
  --input examples/guitar_loop.wav \
  --output outputs/ambient_generated.wav \
  --device cuda
```

### 6-3. 실시간 MIDI 키보드 모드

```bash
# 연결된 MIDI 포트 확인
python -c "import rtmidi; m = rtmidi.MidiIn(); print(m.get_ports())"

# 실행
python main.py --realtime --port "USB MIDI Keyboard" --device cuda
```

### 6-4. 모델·인코더 체크포인트 지정

```bash
python main.py \
  --input examples/theme.mid \
  --model checkpoints/generation/best_model.pt \
  --encoder checkpoints/contrastive/best_encoder.pt \
  --device cuda
```

### 6-5. 전체 CLI 옵션

```
옵션            기본값                          설명
--input, -i     None                           입력 파일 경로 (MIDI/오디오)
--output, -o    None (실시간 재생)              출력 WAV 파일 경로
--model, -m     checkpoints/best_model.pt      생성 모델 체크포인트
--encoder, -e   checkpoints/contrastive/...    인코더 체크포인트
--realtime      False                          실시간 MIDI 입력 모드
--port          None (첫 번째 포트)             MIDI 포트 이름
--device        auto                           cuda / cpu / auto
```

### 6-6. 실행 중 인터랙티브 명령

엔진이 실행되는 동안 다음 명령을 사용할 수 있다:

```
> f examples/new_theme.mid    # 새 파일로 테마 교체
> f                           # 파일 경로 입력 프롬프트
> r                           # 실시간 MIDI 4초 수집 후 교체
> q                           # 종료
```

### 6-7. Python API 직접 사용

```python
from src.streaming.engine import AmbientFlowEngine
import time

engine = AmbientFlowEngine(
    model_ckpt="checkpoints/best_model.pt",
    encoder_ckpt="checkpoints/contrastive/best_encoder.pt",
    output_file="outputs/session.wav",   # 파일로 저장
    device="cuda",
)
engine.start()

# 초기 테마 설정
engine.feed_input(path="inputs/theme1.mid")
time.sleep(30)

# 테마 교체 (재생 중단 없이)
engine.feed_input(path="inputs/theme2.wav")
time.sleep(30)

engine.stop()
```

---

## 7. 훈련 방법

### 7-1. 데이터 준비

**권장 데이터셋:**

| 데이터셋 | 규모 | 용도 |
|---------|------|------|
| Maestro v3 | 피아노 MIDI, ~200시간 | 메인 훈련 |
| GiantMIDI-Piano | 피아노, ~172시간 | 보조 훈련 |
| POP909 | 팝 피아노, 909곡 | 멜로디 패턴 |

```bash
# data/raw_midi/ 에 MIDI 파일 복사 후
python scripts/preprocess_data.py \
  --input data/raw_midi/ \
  --output data_pkl/
```

전처리 파이프라인:
1. 노트 퀀타이즈 (32분음표 그리드)
2. 임시 테마 추출 (랜덤 초기화 인코더 사용)
3. 토크나이즈
4. 512토큰 청크로 분할 (50% overlap)
5. PKL 파일로 저장

### 7-2. Phase 1: 대조 학습 (SegmentEncoder 훈련)

```bash
python training/train_contrastive.py
```

- 설정: `configs/training_config.yaml` → `contrastive` 섹션
- 배치 크기: 256, 학습률: 3e-4, 에폭: 100
- 체크포인트: `checkpoints/contrastive/`
- 소요 시간: GPU 기준 ~2~4시간 (데이터 규모에 따라 다름)

### 7-3. Phase 2: 테마 레이블 재생성

훈련된 인코더로 데이터를 다시 전처리한다:

```bash
python scripts/preprocess_data.py \
  --input data/raw_midi/ \
  --output data_pkl/
# ThemeExtractor가 학습된 checkpoints/contrastive/best_encoder.pt를 자동으로 사용
```

### 7-4. Phase 3: 생성 모델 훈련 (AmbientFlowModel)

```bash
python training/train_generator.py
```

- 설정: `configs/training_config.yaml` → `generation` 섹션
- 배치 크기: 32, 학습률: 1e-4, 에폭: 200
- 체크포인트: `checkpoints/generation/best_model.pt`
- GPU 8GB: ~12~24시간 (Maestro 기준)

### 7-5. Phase 4: 앰비언트 파인튜닝 (선택)

앰비언트 특화 데이터(느린 곡, 패드 중심)만 별도로 모아서 미세 조정:

```bash
# data_pkl/ambient/ 에 앰비언트 PKL 파일 준비 후
# configs/training_config.yaml의 finetune 섹션 설정
python training/train_generator.py  # finetune 설정 사용하도록 수정 필요
```

### 훈련 설정 요약

```yaml
# configs/training_config.yaml

contrastive:
  epochs: 100
  batch_size: 256
  lr: 3.0e-4

generation:
  epochs: 200
  batch_size: 32       # GPU 메모리 부족 시 줄이기
  lr: 1.0e-4
  gradient_clip: 1.0
```

---

## 8. 테스트

### 전체 테스트 실행

```bash
pytest tests/ -v --asyncio-mode=auto
```

**현재 결과: 26/26 통과**

```
tests/test_integration.py::test_full_pipeline_midi_to_theme    PASSED
tests/test_integration.py::test_tokenizer_theme_marker_pipeline PASSED
tests/test_integration.py::test_model_forward_with_theme        PASSED
tests/test_model.py::test_gpa_forward_shape                     PASSED
tests/test_model.py::test_gpa_gate_effect                       PASSED
tests/test_model.py::test_model_forward                         PASSED
tests/test_model.py::test_model_generate_chunk                  PASSED
tests/test_model.py::test_model_parameter_count                 PASSED
tests/test_streaming.py::test_theme_buffer_update_get           PASSED
tests/test_streaming.py::test_theme_buffer_new_flag             PASSED
tests/test_streaming.py::test_theme_buffer_thread_safety        PASSED
tests/test_streaming.py::test_crossfade_shape                   PASSED
tests/test_streaming.py::test_crossfade_short_audio             PASSED
tests/test_streaming.py::test_token_queue_put_get               PASSED
tests/test_theme_extractor.py::test_extractor_returns_tokens    PASSED
tests/test_theme_extractor.py::test_extractor_tokens_valid_ids  PASSED
tests/test_theme_extractor.py::test_contrastive_loss_shape      PASSED
tests/test_theme_extractor.py::test_clustering_returns_candidates PASSED
tests/test_tokenizer.py::test_vocab_size                        PASSED
tests/test_tokenizer.py::test_midi_to_tokens_not_empty          PASSED
tests/test_tokenizer.py::test_tokens_contain_pitch              PASSED
tests/test_tokenizer.py::test_tokens_to_midi_roundtrip          PASSED
tests/test_tokenizer.py::test_theme_markers                     PASSED
tests/test_tokenizer.py::test_encode_decode_roundtrip           PASSED
tests/test_tokenizer.py::test_closest_dur_token                 PASSED
tests/test_tokenizer.py::test_closest_vel_token                 PASSED
```

### 빠른 테스트 (스트리밍 제외)

```bash
pytest tests/ -v -k "not streaming"
```

### 커버리지 리포트

```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
# htmlcov/index.html 에서 확인
```

### 품질 기준 (Definition of Done)

| 단계 | 기준 | 현재 상태 |
|------|------|----------|
| 토크나이저 | 왕복 변환 노트 수 오차 ≤ 10% | 통과 |
| 테마 추출 | Theme Recall@3 ≥ 0.6 (학습 후) | 훈련 필요 |
| 생성 모델 | Validation Perplexity ≤ 3.0 | 훈련 필요 |
| 스트리밍 | 청크 간 갭 ≤ 50ms | 하드웨어 검증 필요 |
| 통합 | 10분 연속 생성 중 크래시 없음 | 훈련 후 검증 필요 |
| 앰비언트 품질 | MOS ≥ 3.5 (5점 척도) | 훈련 후 평가 필요 |

---

## 9. 설정 파일 가이드

### `configs/model_config.yaml` — 모델 크기 조절

```yaml
# Small 설정 (현재 기본값) — CPU 실행 가능
vocab_size: 238
d_model: 256
max_seq_len: 512
decoder_layers: 4
decoder_heads: 4
ff_dim: 1024
gpa_gate_bias: -1.0    # 초기에 테마 영향 약하게 시작

# Base 설정 — GPU 권장
# d_model: 512
# max_seq_len: 1024
# decoder_layers: 8
# decoder_heads: 8
# ff_dim: 2048
```

### `configs/streaming_config.yaml` — 레이턴시 vs 안정성

```yaml
streaming:
  chunk_size: 128      # 줄이면 반응 빠름, 늘리면 품질 향상
  context_len: 512     # 생성 컨텍스트 윈도우 크기
  queue_maxsize: 4     # 선읽기 청크 수 (값이 크면 버퍼 여유 증가)

fluidsynth:
  soundfont: "data/soundfonts/FluidR3_GM.sf2"  # 사운드폰트 경로
  reverb_room: 0.8     # 잔향 크기 (앰비언트는 크게)
  reverb_wet: 0.4      # 잔향 비율
```

### `configs/training_config.yaml` — GPU 메모리 부족 시

```yaml
generation:
  batch_size: 16       # 기본 32 → GPU OOM 시 줄이기
  lr: 1.0e-4
```

---

## 10. 트러블슈팅

| 증상 | 원인 | 해결책 |
|------|------|--------|
| `AttributeError: 'PrettyMIDI' object has no attribute 'get_tempo_change_times'` | pretty_midi 버전 차이 | `get_tempo_changes()` 사용 (이미 적용됨) |
| `crossfade dtype float64` | `np.linspace`가 float64 반환 | `.astype(np.float32)` 명시 (이미 적용됨) |
| FluidSynth 오디오 없음 | 사운드폰트 경로 오류 | `configs/streaming_config.yaml`의 `fluidsynth.soundfont` 확인 |
| CUDA OOM | 배치 크기 과다 | `configs/training_config.yaml`에서 `batch_size` 줄이기 |
| 테마 추출 실패 | 입력이 너무 짧음 (< 4초) | 최소 8초 이상 입력 권장 |
| 생성 청크 갭 | 모델이 너무 느림 | Small 모델 사용 또는 `chunk_size` 줄이기 |
| `RuntimeError: MIDI 입력 포트가 없습니다` | MIDI 장치 미연결 | `python -c "import rtmidi; print(rtmidi.MidiIn().get_ports())"` |
| `ModuleNotFoundError: No module named 'basic_pitch'` | 미설치 | `pip install basic-pitch` |

---

## 11. 구현 현황 및 다음 단계

### 구현 완료 (Phase 1~4)

```
Phase 1: 기반 인프라           ██████████ 완료
Phase 2: 테마 추출 시스템       ██████████ 완료 (훈련 제외)
Phase 3: 생성 모델             ██████████ 완료 (훈련 제외)
Phase 4: 실시간 스트리밍        ██████████ 완료 (하드웨어 검증 제외)
Phase 5: 앰비언트 최적화        ░░░░░░░░░░ 미착수
```

### 다음 단계 (Phase 5)

1. **데이터셋 준비**: Maestro v3 또는 GiantMIDI-Piano 다운로드 후 `data/raw_midi/`에 배치
2. **전처리**: `python scripts/preprocess_data.py`
3. **대조 학습**: `python training/train_contrastive.py`
4. **전처리 재실행**: 학습된 인코더로 테마 레이블 갱신
5. **생성 모델 훈련**: `python training/train_generator.py`
6. **실행 검증**: `python main.py --input examples/test.mid --device cuda`
7. **앰비언트 파인튜닝**: 느린 템포, 패드 위주 데이터로 미세 조정
8. **평가**: Theme Recall@3, Perplexity, MOS 측정

### 의존 관계 (import 트리)

```
main.py
  └── src/streaming/engine.py
        ├── src/input/normalizer.py
        │     ├── src/input/midi_handler.py
        │     └── src/input/audio_handler.py
        ├── src/theme/extractor.py
        │     ├── src/theme/tokenizer.py
        │     ├── src/theme/contrastive.py
        │     └── src/theme/clustering.py
        ├── src/model/transformer.py
        │     ├── src/model/config.py
        │     ├── src/model/gated_attention.py
        │     └── src/model/decoder.py
        ├── src/streaming/buffer.py
        ├── src/streaming/chunk_generator.py
        ├── src/streaming/crossfade.py
        └── src/output/renderer.py
```

순환 의존성 없음 — 위 트리 순서로 구현 및 테스트 완료.
