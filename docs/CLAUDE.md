# CLAUDE.md — Claude Code 작업 지시서

> **이 파일을 먼저 읽으세요.**
> Claude Code가 이 프로젝트에서 작업할 때 따라야 하는 전체 지침과 순서를 정의합니다.

---

## 프로젝트 한 줄 요약

MIDI/오디오 입력에서 테마를 추출하고, 테마를 조건으로 앰비언트 음악을 끊임없이 실시간 생성하는 Python 시스템.

---

## 핵심 참고 문서 (순서대로 읽기)

| 번호 | 파일 | 내용 |
|------|------|------|
| 00 | `docs/00_PROJECT_OVERVIEW.md` | 전체 아키텍처, 컴포넌트 목록, 디렉토리 구조 |
| 01 | `docs/01_ENVIRONMENT_SETUP.md` | 의존성, 설치 명령어, 환경 검증 |
| 02 | `docs/02_MUSIC_REPRESENTATION.md` | 토큰 어휘, 토크나이저 구현 |
| 03 | `docs/03_THEME_EXTRACTOR.md` | 대조 학습 인코더, 클러스터링, 테마 추출기 |
| 04 | `docs/04_MODEL_ARCHITECTURE.md` | GPA 모듈, 디코더, 전체 모델 |
| 05 | `docs/05_STREAMING_ENGINE.md` | 비동기 버퍼, 청크 생성기, 렌더러 |
| 06 | `docs/06_INPUT_HANDLER.md` | MIDI/오디오 입력 핸들러 |
| 07 | `docs/07_TRAINING_PIPELINE.md` | 데이터셋, 손실 함수, 훈련 스크립트 |
| 08 | `docs/08_INTERFACE_AND_MAIN.md` | 메인 엔진, CLI, API |
| 09 | `docs/09_TESTING.md` | 단위/통합 테스트, 품질 기준 |

---

## 구현 순서 (Phase별)

### Phase 1: 기반 인프라 (가장 먼저)
```
1. 디렉토리 구조 생성 (00_PROJECT_OVERVIEW.md 참조)
2. pyproject.toml, requirements.txt 작성
3. REMIAmbientTokenizer 구현 (02_MUSIC_REPRESENTATION.md)
4. MIDIFileHandler, AudioHandler 구현 (06_INPUT_HANDLER.md)
5. 환경 검증 스크립트 작성
6. 테스트: test_tokenizer.py, 기본 MIDI 로드 확인
```

### Phase 2: 테마 추출 시스템
```
1. SegmentEncoder + NTXentLoss 구현 (03_THEME_EXTRACTOR.md)
2. cluster_segments 함수 구현
3. ThemeExtractor 통합 클래스 구현
4. ContrastiveDataset 구현 (07_TRAINING_PIPELINE.md)
5. train_contrastive.py 작성
6. 테스트: test_theme_extractor.py
```

### Phase 3: 생성 모델
```
1. ModelConfig 작성 (04_MODEL_ARCHITECTURE.md)
2. GPAModule 구현 (핵심 기여, 주의 깊게 구현)
3. ThemeEncoder, AmbientMusicDecoder 구현
4. AmbientFlowModel 통합
5. ThemeGenerationDataset, ThemeAwareCrossEntropy 구현
6. train_generator.py 작성
7. 테스트: test_model.py
```

### Phase 4: 실시간 스트리밍
```
1. ThemeBuffer, TokenQueue 구현 (05_STREAMING_ENGINE.md)
2. ChunkGenerator 구현
3. crossfade_audio 구현
4. RealtimeRenderer + FluidSynth 통합
5. AmbientFlowEngine 조율 클래스 구현
6. main.py CLI 구현
7. 테스트: test_streaming.py, test_integration.py
```

### Phase 5: 앰비언트 최적화
```
1. 앰비언트 전용 파인튜닝 스크립트
2. FluidSynth reverb 파라미터 튜닝
3. 생성 온도/top-p 앰비언트 최적화
4. 전체 시스템 테스트 및 문서 정리
```

---

## 작업 시 준수 사항

### ✅ 반드시 지킬 것

1. **각 파일의 위치**: `00_PROJECT_OVERVIEW.md`의 디렉토리 구조를 엄격히 따를 것
2. **타입 힌트**: 모든 함수에 Python 타입 힌트 포함
3. **docstring**: 각 클래스/함수에 한국어 또는 영어 docstring 포함
4. **테스트**: 각 Phase 완료 후 해당 테스트 파일 실행하여 통과 확인
5. **설정 파일**: 하드코딩 금지 — 모든 하이퍼파라미터는 `configs/*.yaml`에서 로드

### ❌ 하지 말 것

1. GPU 없이 실행 불가능한 코드 작성 (항상 `device` 파라미터로 CPU/GPU 선택 가능하게)
2. 절대 경로 하드코딩 (`/home/user/...` 등 금지 — 상대 경로 또는 config 사용)
3. `requirements.txt` 외 패키지 추가 없이 import (추가 시 requirements.txt도 업데이트)
4. 테스트 없이 Phase 완료 선언

---

## 빠른 시작 명령 (구현 완료 후)

```bash
# 1. 환경 설정
conda activate ambientflow
python scripts/check_environment.py

# 2. 데이터 전처리
python scripts/preprocess_data.py --input data/raw_midi/ --output data_pkl/

# 3. Phase 1: 대조 학습 훈련
python training/train_contrastive.py

# 4. Phase 2: 생성 모델 훈련
python training/train_generator.py

# 5. 실행
python main.py --input examples/test.mid --device cuda
```

---

## 트러블슈팅 가이드

| 문제 | 가능한 원인 | 해결책 |
|------|-----------|--------|
| FluidSynth 오디오 없음 | 사운드폰트 경로 오류 | `configs/streaming_config.yaml`의 `fluidsynth.soundfont` 확인 |
| CUDA OOM | 배치 크기 또는 seq_len 과다 | `configs/training_config.yaml`에서 `batch_size` 줄이기 |
| 테마 추출 실패 | 입력이 너무 짧음 (< 4초) | 최소 8초 이상 입력 권장 |
| 생성 청크 갭 | 모델이 너무 느림 | Small 모델 사용 또는 `chunk_size` 줄이기 |
| rtmidi 포트 없음 | MIDI 장치 미연결 | `python -c "import rtmidi; print(rtmidi.MidiIn().get_ports())"` |
| basic-pitch 오류 | 모델 미다운로드 | `pip install basic-pitch --upgrade` |

---

## 의존관계 요약 (import 관계)

```
main.py
  └── src/streaming/engine.py (AmbientFlowEngine)
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
              ├── src/theme/tokenizer.py (재사용)
              └── src/streaming/buffer.py (재사용)
```

**순환 의존성 없음** — 위 트리 순서로 구현하면 의존성 문제 없음.
