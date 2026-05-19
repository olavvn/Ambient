# AmbientFlow: Theme-Aware Real-Time Ambient Music Generator

## 프로젝트 개요

MIDI 또는 오디오 입력을 받아, 해당 입력에서 **테마(주제 선율)**를 추출하고, 이를 조건으로 삼아 **앰비언트 음악을 끊임없이 실시간으로 생성**하는 시스템이다. 중간에 새로운 입력이 들어오면 부드러운 전환(crossfade)을 거쳐 새로운 테마로 반응한다.

### 핵심 아이디어 결합

| 출처 | 핵심 아이디어 | 본 프로젝트 적용 |
|------|--------------|----------------|
| ThemeTransformer | 테마 기반 컨디셔닝 (대조 학습 + 게이티드 병렬 어텐션) | 입력에서 테마를 추출하고, 해당 테마를 반복·변형하며 생성 |
| Magenta RealTime | 실시간 연속 스트리밍 생성, 라이브 유저 컨트롤 | 청크 단위 롤링 생성, 새 입력에 즉각 반응 |
| 본 프로젝트 | 앰비언트 장르 특화 | 느린 템포, 텍스처, 롱레인지 반복, 부드러운 전환 |

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        AmbientFlow System                        │
│                                                                   │
│  ┌──────────────┐    ┌───────────────────┐    ┌──────────────┐  │
│  │ Input Handler│───▶│ Theme Extractor   │───▶│ Theme Buffer │  │
│  │              │    │ (Contrastive +    │    │ (current +   │  │
│  │ MIDI / Audio │    │  Clustering)      │    │  incoming)   │  │
│  └──────────────┘    └───────────────────┘    └──────┬───────┘  │
│                                                       │          │
│  ┌──────────────────────────────────────────────────▼───────┐   │
│  │                  Streaming Generation Engine              │   │
│  │                                                           │   │
│  │   ┌──────────────────────────────────────────────────┐   │   │
│  │   │  Theme-Conditioned Transformer (seq2seq)         │   │   │
│  │   │  ┌─────────────┐      ┌────────────────────────┐ │   │   │
│  │   │  │   Theme     │      │   Music Decoder        │ │   │   │
│  │   │  │   Encoder   │─────▶│   + Gated Parallel     │ │   │   │
│  │   │  │             │      │     Attention Module   │ │   │   │
│  │   │  └─────────────┘      └────────────┬───────────┘ │   │   │
│  │   └────────────────────────────────────┼─────────────┘   │   │
│  │                                         │                 │   │
│  │   Rolling Context Window ◀──────────────┘                 │   │
│  │   (이전 생성 토큰 + 새 청크)                                │   │
│  └───────────────────────────────────┬───────────────────────┘   │
│                                       │                           │
│  ┌────────────────────────────────────▼──────────────────────┐   │
│  │                   Output Stream                            │   │
│  │   Token Sequence ──▶ MIDI Events ──▶ Audio (FluidSynth)   │   │
│  │                         │                                  │   │
│  │                    Crossfade Engine (테마 전환 시)          │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 핵심 컴포넌트

### 1. Input Handler (`src/input/`)
- MIDI 파일 / 실시간 MIDI 포트 / 오디오 파일 입력 수신
- 오디오 → MIDI 변환 (기본음 추출, onset detection)
- 공통 내부 표현(Event Token Sequence)으로 정규화

### 2. Theme Extractor (`src/theme/`)
- **대조 학습(Contrastive Learning)**: 유사 패시지는 임베딩 공간에서 가깝게, 다른 패시지는 멀게
- **클러스터링(K-Means / DBSCAN)**: 임베딩 공간에서 반복 등장하는 패턴 = 테마 후보
- **테마 선택**: 가장 대표적인 클러스터 centroid를 테마로 채택
- 앰비언트 특화: 짧은 멜로디 셀(4~8마디)보다는 텍스처·하모니 패턴을 테마로 선호

### 3. Theme-Conditioned Transformer (`src/model/`)
- **Theme Encoder**: 테마 토큰 시퀀스를 컨텍스트 벡터로 인코딩
- **Gated Parallel Attention (GPA)**: 디코더에서 일반 self-attention과 병렬로 theme-cross-attention 수행, 게이트로 혼합 비율 동적 조절
- **앰비언트 디코더**: 온도(temperature), top-p 샘플링 조절로 부드럽고 예측 가능한 전개

### 4. Streaming Engine (`src/streaming/`)
- **청크 기반 롤링 생성**: N토큰 단위 청크를 반복 생성, 이전 청크를 컨텍스트 윈도우로 유지
- **비동기 큐**: 생성 스레드 / 재생 스레드 분리 (생산자-소비자 패턴)
- **Crossfade Module**: 새 테마 입력 시 현재 청크 종료 지점까지 생성 완료 후 부드럽게 전환

### 5. Output Renderer (`src/output/`)
- Token → MIDI event 변환
- FluidSynth를 통한 실시간 오디오 렌더링
- 앰비언트 특화 후처리: reverb, 다이나믹스 스무딩

---

## 데이터 표현 (Token Vocabulary)

ThemeTransformer의 REMI+ 기반 표현을 앰비언트에 맞게 확장:

```
[BAR]                        # 마디 시작
[TEMPO_BPM_xx]               # 템포 (40~120 BPM, 앰비언트 범위)
[POSITION_x/16]              # 박자 내 위치 (16분음표 단위)
[PITCH_xx]                   # 음고 (0~127)
[DURATION_x/16]              # 음길이
[VELOCITY_x]                 # 세기 (8단계)
[SUSTAIN_ON/OFF]             # 서스테인 페달
[THEME_START] / [THEME_END]  # 테마 구간 마커
[THEME_REF]                  # 현재 생성이 테마를 참조함을 표시
```

---

## 디렉토리 구조

```
ambientflow/
├── docs/                        # 이 마크다운 파일들
├── src/
│   ├── input/
│   │   ├── midi_handler.py      # MIDI 입력 처리
│   │   ├── audio_handler.py     # 오디오 → MIDI 변환
│   │   └── normalizer.py        # 공통 표현 변환
│   ├── theme/
│   │   ├── extractor.py         # 테마 추출 파이프라인
│   │   ├── contrastive.py       # 대조 학습 모델
│   │   ├── clustering.py        # 클러스터링 로직
│   │   └── tokenizer.py         # 테마 토크나이저
│   ├── model/
│   │   ├── theme_encoder.py     # 테마 인코더
│   │   ├── gated_attention.py   # Gated Parallel Attention
│   │   ├── decoder.py           # 앰비언트 뮤직 디코더
│   │   ├── transformer.py       # 전체 seq2seq 모델
│   │   └── config.py            # 모델 하이퍼파라미터
│   ├── streaming/
│   │   ├── engine.py            # 스트리밍 생성 엔진
│   │   ├── chunk_generator.py   # 청크 단위 생성
│   │   ├── crossfade.py         # 테마 전환 크로스페이드
│   │   └── buffer.py            # 비동기 생성/재생 버퍼
│   ├── output/
│   │   ├── renderer.py          # Token → MIDI → Audio
│   │   ├── midi_writer.py       # MIDI 이벤트 작성
│   │   └── postprocess.py       # 앰비언트 후처리
│   └── utils/
│       ├── logger.py
│       └── config_loader.py
├── training/
│   ├── train_contrastive.py     # 대조 학습 훈련
│   ├── train_generator.py       # 생성 모델 훈련
│   ├── dataset.py               # 데이터셋 클래스
│   └── losses.py                # 손실 함수
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── streaming_config.yaml
├── tests/
│   ├── test_theme_extractor.py
│   ├── test_model.py
│   ├── test_streaming.py
│   └── test_integration.py
├── scripts/
│   ├── preprocess_data.py
│   ├── download_dataset.sh
│   └── evaluate.py
├── main.py                      # 메인 진입점
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 구현 우선순위 (Phase)

| Phase | 목표 | 산출물 |
|-------|------|--------|
| 1 | 환경 구축 + 데이터 파이프라인 | 토크나이저, 데이터로더 |
| 2 | 테마 추출기 (대조 학습) | 학습된 임베딩 모델 |
| 3 | Theme-Conditioned Transformer | 기본 생성 모델 |
| 4 | 스트리밍 엔진 | 실시간 청크 생성 |
| 5 | 앰비언트 특화 튜닝 + UI | 완성된 시스템 |
