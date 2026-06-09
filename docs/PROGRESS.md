# AnchorFlow-Simple 구현 진행 현황

## 완료된 작업

### Step 1 — 브랜치
- `feature/anchorflow-lite` 브랜치에서 작업 중

### Step 2 — 의존성
- `requirements.txt`에 transformers, miditok, huggingface_hub, accelerate, torch, pretty_midi 포함 (기존 유지)

### Step 3 — 토크나이저
- `src/tokenizer.py` — TSDTokenizer (274 토큰 어휘, TSD Ambient)
- `src/theme/tokenizer.py`는 하위 호환용으로 유지

### Step 4 — 사전학습 모델 래퍼
- `src/model/anchorflow.py` — AnchorFlowModel, load_pretrained_amt, resize_and_reinit_embeddings

### Step 5 — 자체 구현 모델 제거
- `src/model/transformer.py` 삭제
- `src/model/decoder.py` 삭제
- `src/model/gated_attention.py` 삭제
- `src/model/config.py` 단순화

### Step 6 — 학습 파이프라인
- `training/dataset.py` — AmbientMIDIDataset (슬라이딩 윈도우)
- `training/losses.py` — 표준 CrossEntropy
- `training/train_phase1.py` — 임베딩 적응 (임베딩+LM head만 학습)
- `training/train_phase2.py` — 전체 파인튜닝 (차별화된 학습률)

### Step 7 — 스트리밍 단순화
- `src/streaming/buffer.py` — AnchorQueue + TokenQueue
- `src/streaming/chunk_generator.py` — 상태머신 제거, HF generate() 사용
- `src/streaming/engine.py` — AnchorFlowEngine

### Step 8 — 불필요 파일 제거
- `src/theme/extractor.py`, `contrastive.py`, `clustering.py`, `style_library.py` 제거
- `training/train_contrastive.py`, `train_generator.py` 제거

### Step 9 — 설정 파일 정리
- `configs/model_config.yaml` — pretrained_model_id, vocab_size
- `configs/training_config.yaml` — phase1/phase2 분리
- `configs/streaming_config.yaml` — chunk_size, max_context, sampling

### Step 10 — 다운로드 스크립트
- `scripts/download_pretrained.py`

### Step 11 — 통합 테스트
- `tests/test_pipeline.py` — 7개 테스트 모두 통과 ✅

## 다음 작업

- AMT 가중치 다운로드: `python scripts/download_pretrained.py`
- 학습 데이터 준비: `data/processed/train/*.pkl`, `data/processed/val/*.pkl`
- Phase 1 학습: `python training/train_phase1.py --config configs/training_config.yaml`
- Phase 2 학습: `python training/train_phase2.py --config configs/training_config.yaml --phase1_ckpt checkpoints/phase1_best.pt`
