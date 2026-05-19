# 01 환경 구축 (Environment Setup)

## 시스템 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| OS | Ubuntu 20.04 / macOS 12 | Ubuntu 22.04 |
| Python | 3.9 | 3.10 |
| GPU | NVIDIA 8GB VRAM (훈련) | NVIDIA 24GB VRAM |
| RAM | 16 GB | 32 GB |
| 저장공간 | 50 GB | 200 GB (데이터셋 포함) |

---

## 1. Conda 환경 생성

```bash
conda create -n ambientflow python=3.10 -y
conda activate ambientflow
```

---

## 2. 핵심 의존성 설치

### 2-1. PyTorch (CUDA 12.1 기준)
```bash
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2-2. 음악 처리 라이브러리
```bash
# MIDI 처리
pip install pretty_midi mido python-rtmidi miditoolkit

# 오디오 처리
pip install librosa soundfile pyaudio

# 오디오 → MIDI 변환 (기본음 추출)
pip install basic-pitch

# MIDI 렌더링 (FluidSynth)
sudo apt-get install -y fluidsynth fluid-soundfont-gm
pip install pyfluidsynth

# 오디오 크로스페이드/DSP
pip install scipy pydub
```

### 2-3. 딥러닝 유틸리티
```bash
pip install transformers einops rotary-embedding-torch
pip install scikit-learn umap-learn   # 클러스터링
pip install wandb tensorboard          # 실험 추적
```

### 2-4. 기타
```bash
pip install numpy pandas tqdm pyyaml click rich
pip install pytest pytest-asyncio      # 테스트
```

### 2-5. requirements.txt (전체 고정)
```
# requirements.txt
torch==2.2.0
torchaudio==2.2.0
transformers==4.38.0
einops==0.7.0
rotary-embedding-torch==0.5.3

pretty_midi==0.2.10
mido==1.3.1
python-rtmidi==1.5.8
miditoolkit==1.0.1
librosa==0.10.1
soundfile==0.12.1
pyaudio==0.2.14
basic-pitch==0.2.6
pyfluidsynth==1.3.3

scikit-learn==1.4.0
umap-learn==0.5.5
scipy==1.12.0
pydub==0.25.1

numpy==1.26.4
pandas==2.2.0
tqdm==4.66.2
pyyaml==6.0.1
click==8.1.7
rich==13.7.0
wandb==0.16.3
pytest==8.0.0
pytest-asyncio==0.23.5
```

---

## 3. 외부 레포지토리 세팅

### ThemeTransformer 참조 코드
```bash
git clone https://github.com/atosystem/ThemeTransformer.git references/ThemeTransformer
```

### Magenta RealTime 참조 코드
```bash
git clone https://github.com/magenta/magenta-realtime.git references/magenta-realtime
# 의존성 설치 (참조용, 직접 import는 최소화)
pip install -e references/magenta-realtime
```

---

## 4. 프로젝트 초기화

```bash
git clone <this-repo> ambientflow
cd ambientflow
pip install -e .   # setup.py / pyproject.toml 기반 개발 설치
```

### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "ambientflow"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = []   # requirements.txt 참조

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

---

## 5. 사운드폰트 설정 (FluidSynth)

```bash
# 기본 GM 사운드폰트 경로 확인
ls /usr/share/sounds/sf2/

# 고품질 사운드폰트 다운로드 (선택)
wget -O data/soundfonts/GeneralUser.sf2 \
  "https://github.com/akai-professional/general-user-gs/releases/download/1.471/GeneralUser_GS_v1.471.zip"
```

`configs/streaming_config.yaml`에 경로 설정:
```yaml
fluidsynth:
  soundfont: "data/soundfonts/GeneralUser.sf2"
  reverb: true
  chorus: false
  gain: 0.8
```

---

## 6. 환경 검증

```bash
python scripts/check_environment.py
```

`scripts/check_environment.py` 내용:
```python
"""환경 검증 스크립트"""
import sys

def check():
    checks = []

    # PyTorch + CUDA
    import torch
    cuda_ok = torch.cuda.is_available()
    checks.append(("PyTorch CUDA", cuda_ok, torch.version.cuda))

    # MIDI
    import pretty_midi, mido
    checks.append(("pretty_midi", True, pretty_midi.__version__))

    # Audio
    import librosa
    checks.append(("librosa", True, librosa.__version__))

    # basic-pitch
    import basic_pitch
    checks.append(("basic-pitch", True, "ok"))

    # FluidSynth
    try:
        import fluidsynth
        checks.append(("pyfluidsynth", True, "ok"))
    except Exception as e:
        checks.append(("pyfluidsynth", False, str(e)))

    # scikit-learn
    import sklearn
    checks.append(("scikit-learn", True, sklearn.__version__))

    print("\n=== Environment Check ===")
    for name, ok, detail in checks:
        status = "✅" if ok else "❌"
        print(f"{status} {name}: {detail}")

    failed = [c for c in checks if not c[1]]
    if failed:
        print(f"\n{len(failed)} check(s) failed. Please review setup.")
        sys.exit(1)
    else:
        print("\nAll checks passed! Ready to run AmbientFlow.")

if __name__ == "__main__":
    check()
```

---

## 7. MIDI 포트 확인 (실시간 입력 사용 시)

```bash
python -c "import rtmidi; m = rtmidi.MidiIn(); print(m.get_ports())"
```

DAW 또는 MIDI 컨트롤러 연결 후 포트 이름을 `configs/streaming_config.yaml`의 `midi_input_port`에 설정.
