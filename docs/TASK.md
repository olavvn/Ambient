# AnchorFlow-Simple 아키텍처 설계 문서

> **프로젝트:** 실시간 인터랙티브 앰비언트 음악 생성 시스템
> **코드베이스:** https://github.com/olavvn/Ambient
> **설계 원칙:** 검증된 단순 메커니즘 우선. 신규 아키텍처 발명 회피. 표준 자기회귀 모델 + 토큰 리터럴 삽입으로 모든 요구사항 충족.

---

## 0. 시스템 개요

```
[사용자] ─── MIDI 입력 (anchor) ───┐
                                  ↓
                       ┌──────────────────────────┐
                       │  AnchorFlow Engine       │
                       │  - 디코더 전용 트랜스포머  │
                       │  - 자기회귀 스트리밍       │
                       │  - 컨텍스트에 anchor 삽입  │
                       └──────────────────────────┘
                                  ↓
                          MIDI 토큰 스트림 (실시간)
                                  ↓
                       [출력: 앰비언트 MIDI 재생]
```

**핵심 동작:**

```
t=0   : feed_anchor(A) → context = [BOS, A]
       → 자기회귀 생성 시작 → ambient 토큰 스트림 출력

t=T2  : feed_anchor(B)
       → 다음 청크 경계에서 context에 B 리터럴 삽입
       → context = [..., 생성된_ambient, B]
       → 자기회귀 생성 계속 → 모델이 B의 화성에 자연스럽게 적응

t=T3  : feed_anchor(C)
       → context = [..., 생성된_ambient, C]
       → 계속 생성
```

신규 메커니즘 없음. 표준 자기회귀 + 컨텍스트 조작만으로 모든 요구사항 충족.

---

## 1. 데이터셋과 토크나이저

### 1-1. 학습 데이터셋

**데이터 구축은 별도 트랙으로 진행 중이므로 본 프로젝트 범위 밖이다.** 본 시스템은 토큰화된 앰비언트 MIDI 시퀀스 (`*.pkl` 형식)가 `data/processed/` 폴더에 준비되어 있다고 가정한다.

**데이터 입력 가정:**
```
data/processed/
├── train/
│   ├── ambient_001.pkl   # {"tokens": [int, int, ...]}
│   ├── ambient_002.pkl
│   └── ...
└── val/
    ├── ...
```

**중요:** 데이터에 anchor/non-anchor 구분이나 특수 마킹을 추가하지 않는다. 앰비언트 MIDI 시퀀스를 **그대로 학습**한다. 모델은 데이터 내에서 자연스럽게 발생하는 화성 변화, 텍스처 전환을 학습한다. 추론 시 사용자가 삽입하는 anchor도 데이터에서 본 화성 변화 패턴의 한 형태로 모델은 인식하게 된다.

### 1-2. 토크나이저: TSD Ambient (별도 보고서 설계 그대로 사용)

**핵심 사양:**

```python
from miditok import TSD, TokenizerConfig

AMBIENT_CONFIG = TokenizerConfig(
    pitch_range=(21, 108),                        # A0~C8
    beat_res={(0, 8): 8, (8, 64): 4},             # 긴 노트 대응
    use_tempos=True,
    use_velocities=True,
    use_rests=True,                               # 침묵 명시
    tempo_range=(20, 80),                         # 앰비언트 실제 범위
    nb_tempos=30,
    nb_velocities=16,                             # pp~mf 미세 표현
    special_tokens=["PAD", "BOS", "EOS", "MASK"], # 4개만
    additional_params={
        "max_duration": (32, 0, 8),               # 최대 32박자 (~8마디)
    }
)
```

**앰비언트 특화 CC 처리 (커스텀 확장):**

| CC 번호 | 이름 | 양자화 단계 | 비고 |
|---|---|---|---|
| CC7 | Channel Volume | 16단계 | 페이드 인/아웃 |
| CC11 | Expression | 16단계 | 실시간 음량 표현 |
| CC74 | Brightness | 16단계 | 필터 컷오프 (앰비언트 핵심) |
| CC91 | Reverb Depth | 16단계 | 공간감 (앰비언트 핵심) |
| CC93 | Chorus Depth | 8단계 | 코러스 효과 |

**총 어휘 크기:** 약 382 토큰

**선택 근거 (별도 보고서 참조):**
TSD의 TimeShift 기반 시간 표현은 앰비언트의 비박자적, 유동적 시간 흐름에 REMI의 Bar+Position보다 적합하다. 명시적 Duration 토큰은 긴 패드 음표를 NoteOff 방식보다 안정적으로 표현한다.

### 1-3. 학습 시퀀스 구성

**기존 코드의 ThemeGenerationDataset 대체:**

```python
class AmbientMIDIDataset(Dataset):
    """
    표준 자기회귀 학습 데이터셋.

    앰비언트 MIDI를 토큰화한 후, 슬라이딩 윈도우로 고정 길이 시퀀스를 생성한다.
    특별한 anchor 마킹 없음. 데이터 그대로 학습.
    """

    def __init__(self, data_dir, seq_len=2048, stride=1024, pad_id=0, bos_id=1):
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.samples = []

        for pkl_path in Path(data_dir).glob("*.pkl"):
            tokens = pickle.load(open(pkl_path, "rb"))["tokens"]

            # 곡 시작에 BOS 추가
            tokens = [self.bos_id] + list(tokens)

            # 슬라이딩 윈도우로 분할
            for start in range(0, max(1, len(tokens) - seq_len), stride):
                chunk = tokens[start:start + seq_len]
                if len(chunk) < seq_len // 2:
                    continue
                self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # 패딩
        if len(tokens) < self.seq_len:
            tokens = tokens + [self.pad_id] * (self.seq_len - len(tokens))

        tokens = torch.tensor(tokens, dtype=torch.long)
        return {
            "input_ids":  tokens[:-1],
            "labels":     tokens[1:],
        }
```

**시퀀스 길이 권장값:**
- `seq_len = 2048` (앰비언트 약 30~60초 분량)
- 학습 자원에 여유가 있다면 4096까지 확장 가능
- 더 긴 컨텍스트는 더 자연스러운 장기 일관성을 가져옴

---

## 2. 모델 아키텍처와 학습 방법

### 2-1. 기본 전략: 사전학습 모델 파인튜닝

**선택한 사전학습 모델: Anticipatory Music Transformer (AMT)**

```
체크포인트: stanford-crfm/music-medium-800k
저장소:    https://huggingface.co/stanford-crfm/music-medium-800k
구조:      GPT-2 스타일 디코더 전용 트랜스포머
파라미터:  약 360M
사전학습:  Lakh MIDI 데이터셋 (800K 스텝)
라이선스:  Apache 2.0
```

**왜 AMT인가:**

| 항목 | 근거 |
|---|---|
| 아키텍처 부합 | 우리 시스템과 동일한 디코더 전용 자기회귀 구조 |
| 학습 데이터 규모 | Lakh MIDI 전체 학습 → 음악적 일반 표현 풍부 |
| 공개 체크포인트 | HuggingFace에서 `AutoModelForCausalLM`으로 즉시 로드 |
| 라이선스 | Apache 2.0 (상업적 사용 가능) |
| 시나리오 호환성 | AMT는 사전학습 시 미래 제어 토큰 인터리빙을 학습 → 우리의 anchor 삽입 시나리오와 자연스럽게 호환 |

**참고:** 우리는 AMT의 anticipation 메커니즘(미래 제어 토큰 인터리빙)을 사용하지 않는다. 사전학습된 가중치만 가져와서 표준 next-token prediction으로 파인튜닝한다.

### 2-2. 토크나이저 교체 전략 (옵션 B)

**선택:** TSD Ambient 토크나이저로 교체. AMT의 자체 토큰 포맷을 사용하지 않는다.

**의미:**

```
유지: 트랜스포머 블록 가중치 (attention, FFN, LayerNorm)
재초기화: 토큰 임베딩 레이어, LM head (vocab_size 변경: AMT → ~382)
```

**의식해야 할 trade-off:**

AMT의 사전학습 가중치는 자체 토큰 포맷에 기반하여 800K 스텝 학습되었다. 임베딩 레이어를 새 토크나이저로 교체하면, 트랜스포머 블록이 학습한 "어떤 임베딩 패턴이 어떤 음악적 의미를 갖는지"에 대한 표현이 처음에는 일치하지 않는다. 결과적으로:

- **유지되는 능력:** 장기 의존성 모델링, 시퀀스 패턴 인식, 어텐션 분포 학습
- **재학습 필요:** 토큰별 의미 (특정 임베딩이 어떤 음표/CC를 의미하는지)

이 단절을 단계적 파인튜닝으로 완화한다.

### 2-3. 모델 구조 변경

```python
from transformers import AutoModelForCausalLM, AutoConfig

# 1단계: AMT 사전학습 가중치 로드
base_model = AutoModelForCausalLM.from_pretrained(
    'stanford-crfm/music-medium-800k'
)

# 2단계: 어휘 크기를 TSD Ambient에 맞게 조정
new_vocab_size = ambient_tokenizer.vocab_size  # ~382

# resize_token_embeddings로 임베딩/LM head 교체
# (기존 트랜스포머 블록 가중치는 그대로 유지됨)
base_model.resize_token_embeddings(new_vocab_size)

# 3단계: 임베딩 레이어와 LM head를 의도적으로 재초기화
# (resize_token_embeddings는 기존 임베딩 일부를 보존하려 시도하지만,
#  토큰 의미가 완전히 다르므로 깨끗한 재초기화가 더 안전)
import torch.nn as nn
nn.init.normal_(base_model.get_input_embeddings().weight, mean=0.0, std=0.02)
nn.init.normal_(base_model.get_output_embeddings().weight, mean=0.0, std=0.02)
```

### 2-4. 학습 방법: 두 단계 파인튜닝

**Phase 1 — 임베딩 적응 (3~5 에폭)**

```
목적: 새 토크나이저의 임베딩 공간을 트랜스포머 블록의 표현 공간에 정렬
동결: 트랜스포머 블록 전체 (attention, FFN, LayerNorm)
학습: 토큰 임베딩 레이어 + LM head만
학습률: 1e-3 (상대적으로 높게)
이유: 임베딩만 빠르게 적응시켜 블록의 기존 표현을 활용 가능하도록
```

```python
# Phase 1 설정
for name, param in model.named_parameters():
    if 'wte' in name or 'lm_head' in name or 'wpe' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3,
    weight_decay=0.01,
)
```

**Phase 2 — 전체 파인튜닝 (20~30 에폭)**

```
목적: 앰비언트 분포로의 전면적 적응
동결: 없음 (모든 파라미터 학습)
학습률 차별화:
  - 임베딩 + LM head: 1e-4
  - 트랜스포머 블록:    1e-5
이유: 트랜스포머 블록은 사전학습된 일반적 음악 표현을 가지고 있으므로
      낮은 학습률로 미세 조정. catastrophic forgetting 방지.
```

```python
# Phase 2 설정 — 학습률 그룹 분리
embedding_params = [p for n, p in model.named_parameters()
                    if 'wte' in n or 'lm_head' in n or 'wpe' in n]
backbone_params = [p for n, p in model.named_parameters()
                   if 'wte' not in n and 'lm_head' not in n and 'wpe' not in n]

optimizer = AdamW(
    [
        {'params': embedding_params, 'lr': 1e-4},
        {'params': backbone_params,  'lr': 1e-5},
    ],
    weight_decay=0.01,
)
```

**학문적 근거:**

이 단계적 파인튜닝 전략은 Howard & Ruder의 ULMFiT (ACL 2018) 및 후속 transfer learning 연구에서 확립된 관행이다. ULMFiT은 사전학습된 언어 모델을 새 도메인에 적응시킬 때 (1) 임베딩 우선 적응 (2) 점진적 해동 (3) 차별화된 학습률이라는 세 가지 원칙을 제시했다. 본 설계는 이 원칙을 우리 상황(임베딩 완전 교체)에 맞게 적용한 것이다.

### 2-5. 학습 시퀀스 구성

```python
class AmbientMIDIDataset(Dataset):
    """표준 자기회귀 학습 데이터셋. 슬라이딩 윈도우."""

    def __init__(self, data_dir, seq_len=2048, stride=1024, pad_id=0, bos_id=1):
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.samples = []

        for pkl_path in Path(data_dir).glob("*.pkl"):
            tokens = pickle.load(open(pkl_path, "rb"))["tokens"]
            tokens = [self.bos_id] + list(tokens)

            for start in range(0, max(1, len(tokens) - seq_len), stride):
                chunk = tokens[start:start + seq_len]
                if len(chunk) < seq_len // 2:
                    continue
                self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        if len(tokens) < self.seq_len:
            tokens = tokens + [self.pad_id] * (self.seq_len - len(tokens))
        tokens = torch.tensor(tokens, dtype=torch.long)
        return {
            "input_ids": tokens[:-1],
            "labels":    tokens[1:],
        }
```

### 2-6. 손실 함수: 표준 Cross-Entropy

커스텀 손실 없음. 표준 next-token prediction.

```python
loss_fn = nn.CrossEntropyLoss(
    ignore_index=pad_id,
    label_smoothing=0.1,
)
```

### 2-7. 학습 설정 권장값

```yaml
pretrained:
  base_model: "stanford-crfm/music-medium-800k"
  resize_embeddings: true
  reinit_embeddings: true   # 의도적 재초기화

phase1:
  description: "임베딩 적응"
  epochs: 5
  lr: 1.0e-3
  freeze_backbone: true
  batch_size: 16
  warmup_steps: 500

phase2:
  description: "전체 파인튜닝"
  epochs: 25
  lr_embedding: 1.0e-4
  lr_backbone: 1.0e-5
  freeze_backbone: false
  batch_size: 8           # 메모리 고려 (전체 학습은 더 무거움)
  gradient_accumulation: 4
  warmup_steps: 1000

common:
  optimizer: AdamW
  weight_decay: 0.01
  betas: [0.9, 0.95]
  lr_schedule: cosine
  gradient_clip: 1.0
  label_smoothing: 0.1
  seq_len: 2048
```

### 2-8. 왜 이 단순한 방식이 작동하는가

자기회귀 모델은 컨텍스트에 있는 토큰들의 분포에 자연스럽게 조건화된다. AMT는 이미 Lakh MIDI로 음악 시퀀스의 일반적 패턴을 학습했다. 우리가 앰비언트 데이터로 파인튜닝하면, 모델은 다음을 학습한다.

```
[앰비언트 구간 1] → [화성 변화 지점] → [앰비언트 구간 2 (새 화성에 적응)]
```

추론 시 사용자가 anchor_B를 컨텍스트에 직접 삽입하는 것은, 학습 시 본 "화성 변화 지점"이 등장하는 상황과 본질적으로 같다. 모델은 anchor_B의 음표, 화성, 음역대를 보고 그에 맞춰 다음 토큰을 생성한다.

**Anchor의 리터럴 등장 보장:**
anchor 토큰들이 컨텍스트에 그대로 삽입되고, 동시에 출력 스트림에도 그대로 보내진다. 모델이 anchor 토큰을 "생성"하는 것이 아니라 시스템이 직접 삽입하므로, anchor가 완전한 형태로 등장하는 것이 100% 보장된다.

### 2-9. 학문적 근거

이 접근은 다음 표준 연구들의 결합이다:

| 적용 기법 | 출처 |
|---|---|
| 사전학습 모델 활용 | Anticipatory Music Transformer (Thickstun et al., ICLR 2024) |
| 디코더 전용 + next-token prediction | GPT-2 (Radford et al., 2019), Music Transformer (Huang et al., ICLR 2019) |
| Priming continuation을 통한 사용자 제어 | Music Transformer |
| 단계적 파인튜닝 + 차별화된 학습률 | ULMFiT (Howard & Ruder, ACL 2018) |
| TSD 토큰화 | MiDiTok 라이브러리, 별도 토크나이저 보고서 |

---

## 3. 실시간 스트리밍 및 실시간 인풋 처리 방식

### 3-1. 핵심 개념: Anchor

**정의:** Anchor는 사용자가 시스템에 제공하는 MIDI 시퀀스. 출력 스트림에 리터럴(literal)로 등장하며, 이후 생성되는 ambient 토큰의 화성적/음색적 컨텍스트를 결정한다.

**Anchor의 성질:**
- 임의의 시점에 임의의 개수로 제공 가능
- 추론 시작 전 사전 등록 불필요
- 각 anchor는 완전한 형태로 출력에 등장 (모델이 "재생성"하지 않음)
- 첫 번째 anchor는 곡의 시작점 역할
- 이후 anchor는 화성 변화 지점 역할

### 3-2. 컴포넌트 구조

```
┌─────────────────────────────────────────────────────────┐
│ AnchorFlowEngine                                        │
│                                                         │
│  ┌──────────────┐                                       │
│  │ AnchorQueue  │  ← feed_anchor() 호출 시 토큰 추가    │
│  └──────────────┘                                       │
│         │                                               │
│         │ 매 청크 경계에서 확인                          │
│         ↓                                               │
│  ┌──────────────────────────────────────┐               │
│  │ ChunkGenerator (자기회귀 루프)        │               │
│  │  1. AnchorQueue 확인                  │               │
│  │     → 있으면 context에 리터럴 삽입    │               │
│  │     → 동시에 TokenQueue로 송출        │               │
│  │  2. 다음 청크 자기회귀 생성           │               │
│  │     → context 확장                    │               │
│  │     → TokenQueue로 송출               │               │
│  └──────────────────────────────────────┘               │
│         │                                               │
│         ↓                                               │
│  ┌──────────────┐                                       │
│  │ TokenQueue   │  → Output Renderer (MIDI/Audio)       │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

### 3-3. AnchorQueue (단순 큐)

```python
import queue
import threading

class AnchorQueue:
    """
    사용자가 입력한 anchor 토큰들의 FIFO 큐.
    feed_anchor()로 추가, ChunkGenerator가 다음 청크 시작 시 소비.
    """

    def __init__(self):
        self._queue = queue.Queue()

    def push(self, tokens: list[int]):
        """사용자 입력 anchor를 큐에 추가 (thread-safe)"""
        self._queue.put(tokens)

    def pop_all(self) -> list[list[int]]:
        """대기 중인 모든 anchor를 한 번에 가져옴"""
        result = []
        while True:
            try:
                result.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return result

    def has_pending(self) -> bool:
        return not self._queue.empty()
```

기존 코드의 `DualAnchorBuffer`, `ThemeBuffer`, 상태머신을 모두 이 단순 큐로 대체.

### 3-4. ChunkGenerator (자기회귀 스트리밍 루프)

```python
class ChunkGenerator:
    """
    자기회귀 청크 생성 + anchor 리터럴 삽입.
    상태머신 없음. 단순 무한 루프.
    """

    def __init__(self, model, tokenizer, anchor_queue, token_queue,
                 chunk_size=64, max_context=2048, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.anchor_queue = anchor_queue
        self.token_queue = token_queue
        self.chunk_size = chunk_size
        self.max_context = max_context
        self.device = device

        self.context = torch.tensor(
            [[self.tokenizer.bos_id]], dtype=torch.long, device=device
        )
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def _loop(self):
        while self._running:
            # ── 1단계: 대기 중인 anchor 처리 ────────────────────
            pending_anchors = self.anchor_queue.pop_all()
            for anchor_tokens in pending_anchors:
                anchor_tensor = torch.tensor(
                    [anchor_tokens], dtype=torch.long, device=self.device
                )
                # context에 리터럴 삽입
                self.context = torch.cat([self.context, anchor_tensor], dim=1)
                # 스트림에도 그대로 송출 (생성 토큰이 아닌 사용자 입력)
                self.token_queue.put(anchor_tokens)

            # ── 2단계: 다음 청크 자기회귀 생성 ──────────────────
            new_tokens = self._generate_chunk(self.chunk_size)
            self.context = torch.cat([self.context, new_tokens], dim=1)
            self.token_queue.put(new_tokens[0].tolist())

            # ── 3단계: context 길이 관리 ────────────────────────
            self._trim_context()

    @torch.no_grad()
    def _generate_chunk(self, n_tokens: int) -> torch.Tensor:
        """KV-cache 활용 자기회귀 생성"""
        generated = self.model.generate(
            self.context,
            max_new_tokens=n_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_id,
        )
        # 새로 생성된 부분만 반환
        return generated[:, self.context.size(1):]

    def _trim_context(self):
        """컨텍스트가 max_context를 초과하면 앞쪽 잘라냄"""
        if self.context.size(1) > self.max_context:
            keep = self.max_context - self.chunk_size * 2
            self.context = self.context[:, -keep:]
```

### 3-5. AnchorFlowEngine (최종 인터페이스)

```python
class AnchorFlowEngine:
    """단일 공개 인터페이스: feed_anchor() + start() + stop()"""

    def __init__(self, model_ckpt: str, config_path: str):
        self.config = load_config(config_path)
        self.tokenizer = TSDAmbientTokenizer.from_config(self.config)
        self.model = AnchorFlowModel.from_checkpoint(model_ckpt)
        self.model.eval()

        self.anchor_queue = AnchorQueue()
        self.token_queue = TokenQueue()  # 출력 큐 (consumer가 읽음)
        self.chunk_gen = ChunkGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            anchor_queue=self.anchor_queue,
            token_queue=self.token_queue,
            chunk_size=self.config.streaming.chunk_size,
            max_context=self.config.streaming.max_context,
        )

    def feed_anchor(self, midi_path: str):
        """
        사용자 MIDI 입력을 anchor로 등록.
        호출 즉시 반환 (실제 삽입은 다음 청크 경계에서).

        시점:
        - 첫 호출 (t=0): 곡의 시작점이 됨
        - 이후 호출: 화성 변화 지점이 됨
        """
        midi = pretty_midi.PrettyMIDI(midi_path)
        tokens = self.tokenizer.midi_to_tokens(midi)
        self.anchor_queue.push(tokens)
        print(f"[AnchorFlow] Anchor queued: {len(tokens)} tokens")

    def start(self):
        """생성 시작. 최소 1개 이상의 anchor가 큐에 있어야 함."""
        if not self.anchor_queue.has_pending():
            raise RuntimeError("최소 1개의 anchor가 필요합니다. feed_anchor()를 먼저 호출하세요.")
        self.chunk_gen.start()

    def stop(self):
        self.chunk_gen.stop()

    def get_output_tokens(self, timeout=0.1) -> list[int] | None:
        """출력 토큰 가져오기 (consumer에서 호출)"""
        try:
            return self.token_queue.get(timeout=timeout)
        except queue.Empty:
            return None
```

### 3-6. 사용 예시

```python
# 시스템 초기화
engine = AnchorFlowEngine(
    model_ckpt="checkpoints/anchorflow_best.pt",
    config_path="configs/inference.yaml",
)

# 첫 anchor 등록 후 생성 시작
engine.feed_anchor("inputs/intro_chord.mid")
engine.start()

# Output renderer 스레드에서 토큰 소비 → MIDI/오디오 재생
# ...

# 시간이 지난 후 (예: 30초 후)
engine.feed_anchor("inputs/mood_change.mid")
# → 자동으로 다음 청크 경계에서 context에 삽입됨
# → 그 이후 생성되는 ambient는 새 anchor에 자연스럽게 적응

# 또 다른 anchor (예: 60초 후)
engine.feed_anchor("inputs/finale.mid")

# 종료
engine.stop()
```

### 3-7. 스트리밍 설정

```yaml
streaming:
  chunk_size: 64           # 청크당 생성 토큰 수
  max_context: 2048        # 자기회귀 컨텍스트 길이
  temperature: 1.0
  top_p: 0.9
  top_k: 0                 # 0 = 비활성
```

**chunk_size 선택:**
- 작을수록 (32~64): anchor 도착 반응성 좋음, KV-cache 효율 낮음
- 클수록 (128~256): anchor 반응 지연 증가, 생성 효율 좋음
- 권장: 64 (앰비언트의 느린 변화 특성상 약간의 지연 허용 가능)

**max_context 선택:**
- 길수록: 더 자연스러운 장기 일관성, 메모리 사용 증가
- 권장: 2048 (앰비언트 약 30~60초)

---

## 4. 기존 코드베이스 변경 계획

### 4-1. 제거 (Remove)

```
src/theme/                          전체 폴더 제거
├── extractor.py                    (자동 추출 불필요)
├── contrastive.py                  (대조 학습 불필요)
├── clustering.py                   (DBSCAN 불필요)
└── style_library.py                (스타일 블렌딩 불필요)

src/model/transformer.py            (자체 구현 모델 → HF AMT 사용)
src/model/decoder.py                (자체 구현 디코더 → HF AMT 사용)
src/model/gated_attention.py        (cross-attention 불필요)

training/train_contrastive.py       (Phase 1 불필요)
```

### 4-2. 수정 (Modify)

| 파일 | 변경 내용 | 변경 규모 |
|---|---|---|
| `src/model/anchorflow.py` (신규) | HF AMT 로드 + 임베딩 교체 래퍼 | 🔴 신규 작성 |
| `src/model/config.py` | 사전학습 모델명, 어휘 크기 설정만 유지 | 🟡 중간 |
| `src/streaming/buffer.py` | ThemeBuffer → AnchorQueue (대폭 단순화) | 🟡 중간 |
| `src/streaming/chunk_generator.py` | HF `model.generate()` 사용 + 상태머신 제거 | 🟡 중간 |
| `src/streaming/engine.py` | AnchorFlowEngine 단일 인터페이스 | 🟡 중간 |
| `src/tokenizer.py` (신규) | TSD Ambient (이전 토크나이저 보고서 참조) | 🔴 신규 작성 |
| `src/theme/tokenizer.py` | 제거 (위치 이동) | — |
| `training/dataset.py` | AmbientMIDIDataset (표준 슬라이딩 윈도우) | 🔴 전면 재작성 |
| `training/losses.py` | 표준 CrossEntropyLoss만 사용 | 🟢 소규모 |
| `training/train_phase1.py` (신규) | 임베딩 적응 단계 | 🔴 신규 작성 |
| `training/train_phase2.py` (신규) | 전체 파인튜닝 단계 | 🔴 신규 작성 |
| `configs/model_config.yaml` | 사전학습 모델 설정 | 🟡 중간 |
| `configs/training_config.yaml` | Phase 1 / Phase 2 분리 설정 | 🔴 전면 재작성 |
| `configs/streaming_config.yaml` | anchor_queue 관련 설정 | 🟢 소규모 |

### 4-3. 유지 (Keep)

```
src/input/                          입력 처리 그대로
src/output/                         출력 렌더링 그대로
src/streaming/crossfade.py          equal-power 크로스페이드 그대로
src/utils/                          유틸리티 그대로
```

### 4-4. Claude Code 작업 순서

```
[Step 1] 브랜치 생성
git checkout -b feature/anchorflow-simple

[Step 2] 의존성 추가
pyproject.toml 또는 requirements.txt에 추가:
  - transformers>=4.40
  - miditok>=3.0
  - huggingface_hub
  - accelerate
  - torch>=2.0
  - pretty_midi
  - mir_eval (평가용)

[Step 3] 토크나이저 구현
src/tokenizer.py에 TSD Ambient 토크나이저 구현
(별도 토크나이저 보고서 참조)
configs/tokenizer_config.yaml 생성

[Step 4] 사전학습 모델 래퍼 작성
src/model/anchorflow.py 신규 작성:
  - load_pretrained_amt(): HF에서 AMT 체크포인트 로드
  - resize_and_reinit_embeddings(): 임베딩 + LM head 교체
  - AnchorFlowModel: 통합 래퍼 클래스

[Step 5] 자체 구현 모델 제거
src/model/transformer.py, decoder.py, gated_attention.py 삭제
src/model/config.py를 사전학습 설정만 남기고 단순화

[Step 6] 학습 파이프라인 구성
training/dataset.py를 AmbientMIDIDataset으로 재작성
training/losses.py를 표준 CrossEntropy로 단순화
training/train_phase1.py 작성 (임베딩 적응)
training/train_phase2.py 작성 (전체 파인튜닝)

[Step 7] 스트리밍 단순화
src/streaming/buffer.py를 AnchorQueue로 재작성
src/streaming/chunk_generator.py:
  - 상태머신 제거
  - HF model.generate() 사용
  - past_key_values로 KV-cache 활용
src/streaming/engine.py를 AnchorFlowEngine로 재작성

[Step 8] 불필요 파일 제거
src/theme/ 폴더 전체 제거
training/train_contrastive.py 제거

[Step 9] 설정 파일 정리
configs/model_config.yaml: pretrained_model_id, vocab_size 등
configs/training_config.yaml: phase1/phase2 분리
configs/streaming_config.yaml: chunk_size, max_context, sampling 파라미터

[Step 10] 가중치 다운로드 스크립트
scripts/download_pretrained.py 작성:
  - stanford-crfm/music-medium-800k 다운로드
  - 캐시 경로 검증

[Step 11] 통합 테스트
tests/test_pipeline.py:
  - 사전학습 모델 로드 + 임베딩 교체 검증
  - 더미 anchor 입력 → 토큰 출력 검증
  - Phase 1 학습 1 스텝 실행
  - Phase 2 학습 1 스텝 실행
```

---

## 5. 전후 비교

| 항목 | 기존 (Theme Transformer 기반) | 신규 (AnchorFlow-Simple) |
|---|---|---|
| 아키텍처 | 인코더-디코더 + GPA (자체 구현) | 디코더 전용 (사전학습 AMT 활용) |
| 신규 어텐션 모듈 | GPAModule | 없음 (HF 모델 그대로) |
| 학습 손실 | ThemeAwareCrossEntropy | 표준 CrossEntropy |
| 학습 데이터 구성 | (theme, target) 쌍 | 앰비언트 MIDI 전체 시퀀스 |
| 특수 토큰 | THEME_S, THEME_E, THEME_REF 등 8개 | BOS, EOS, PAD, MASK 4개 |
| 사전학습 활용 | 없음 (from scratch) | AMT (Lakh MIDI 800K 스텝) |
| 학습 단계 | 단일 | Phase 1 (임베딩 적응) + Phase 2 (전체 파인튜닝) |
| 대조 학습 | 필요 (Phase 1) | 불필요 |
| Anchor 처리 | 단일 ThemeBuffer | 단순 FIFO Queue |
| 상태 관리 | 복잡 (3상태 머신) | 없음 (단순 큐) |
| 학문적 근거 | 조합 설계 | AMT + Music Transformer + ULMFiT 표준 |
| 코드 복잡도 | 높음 | 낮음 |
| 디버깅 용이성 | 낮음 | 높음 |
| 예상 학습 데이터 요구량 | 100~200시간 (from scratch) | 30~100시간 (파인튜닝) |

---

## 6. 학문적 근거 정리

본 설계는 **새로운 아키텍처를 발명하지 않는다.** 다음 표준 연구들의 기법을 그대로 조합한다.

| 적용 기법 | 출처 논문 | 활용 방식 |
|---|---|---|
| 사전학습된 음악 디코더 모델 | Anticipatory Music Transformer (Thickstun et al., ICLR 2024) | 기본 가중치 (stanford-crfm/music-medium-800k) |
| 디코더 전용 트랜스포머 + next-token prediction | GPT-2 (Radford et al., 2019) | 모델 아키텍처 기반 |
| MIDI 토큰 시퀀스 자기회귀 생성 | Music Transformer (Huang et al., ICLR 2019) | 음악 도메인 적용 패러다임 |
| Priming continuation을 통한 사용자 제어 | Music Transformer | Anchor 삽입 메커니즘 근거 |
| 단계적 파인튜닝 + 차별화된 학습률 | ULMFiT (Howard & Ruder, ACL 2018) | Phase 1 → Phase 2 학습 전략 |
| 사전학습 모델의 어휘 확장/교체 | T5 (Raffel et al., 2020), 일반적 transfer learning 관행 | resize_token_embeddings + 재초기화 |
| TSD 토큰화 | MiDiTok 라이브러리 | 토크나이저 구현 |

특히 다음 두 가지가 본 설계의 핵심 학문적 기반이다.

**Music Transformer의 priming continuation:**
사용자가 제공한 priming sequence를 컨텍스트에 두고 자기회귀 생성을 통해 continuation을 만드는 방식. 우리의 anchor 삽입 메커니즘은 이 priming continuation을 **여러 번 반복**하는 것일 뿐이다. 새로운 메커니즘이 아니다.

**ULMFiT의 단계적 파인튜닝 원칙:**
사전학습된 언어 모델을 새 도메인에 적응시킬 때 (1) 임베딩 우선 적응 (2) 점진적 해동 (3) 차별화된 학습률을 적용. 우리는 임베딩을 완전히 교체해야 하는 더 극단적 상황이므로, Phase 1에서 트랜스포머 블록을 완전 동결하고 임베딩만 학습하는 방식으로 이 원칙을 강화 적용한다.

---

## 7. 한계 및 향후 검토 사항

**한계 1: 임베딩 교체에 따른 사전학습 효과 일부 손실**
토큰 임베딩과 LM head를 재초기화하므로, AMT 사전학습 가중치의 일부 효과가 손실된다. 트랜스포머 블록의 일반적 음악 패턴 인식 능력은 보존되지만, "특정 토큰 → 특정 음악적 의미" 매핑은 재학습이 필요하다. Phase 1의 임베딩 적응이 이 단절을 줄이지만, 완전히 제거할 수는 없다.

**완화 방안:**
파인튜닝 데이터가 충분히 확보되면 (50시간 이상) 이 효과는 자연스럽게 보상된다. 데이터가 부족할 경우 옵션 A (AMT 토크나이저 사용)로 폴백할 수 있다.

**한계 2: 장기 일관성**
컨텍스트 길이가 2048 토큰으로 제한되면, anchor_A로부터 멀어진 시점에서 anchor_A의 영향력이 약해진다. 이것은 AMT를 포함한 모든 표준 GPT 모델의 일반적 한계이다.

**완화 방안:**
- 컨텍스트 길이 확장 (AMT는 2048 토큰 기본, 더 큰 변형은 4096 지원)
- Sliding window with overlap
- 추론 시 anchor를 주기적으로 다시 삽입

**한계 3: 화성 단절 가능성**
anchor_B가 anchor_A와 화성적으로 매우 멀 경우, 도착 시점에 단절감이 발생할 수 있다. 자기회귀 생성만으로는 anchor_B 도착 전 미리 그쪽으로 수렴할 수 없다.

이것이 실제 검증에서 문제가 된다면 향후 다음 옵션을 고려:
- **옵션 A:** anchor_B 도착 직후 짧은 크로스페이드 적용 (기존 `crossfade.py` 활용)
- **옵션 B:** UI에서 사용자가 anchor 도착 직전 "예고 신호"를 줄 수 있게 함
- **옵션 C:** 더 복잡한 메커니즘 도입 (현재로서는 권장하지 않음)

**한계 4: 컨텍스트 누적에 따른 메모리 사용**
실시간 스트리밍이 길어질수록 context가 커지므로 KV-cache가 누적된다. `_trim_context()`로 관리하되, trim 시점에서 화성 정보가 일부 손실될 수 있음을 인지해야 한다.

---

## 8. 핵심 메시지

**검증된 사전학습 모델 + 단순한 메커니즘의 조합으로 요구사항이 충족된다:**

```
요구사항 1: Anchor가 완전한 형태로 출력에 등장
  → context에 토큰을 그대로 삽입 + 동시에 출력 큐에 송출

요구사항 2: Anchor 사이를 자연스럽게 앰비언트로 채움
  → AMT 사전학습 가중치 + 앰비언트 데이터 파인튜닝
  → 표준 자기회귀 생성

요구사항 3: 실시간 중간 삽입
  → 청크 경계에서 anchor 큐 확인 → 있으면 context에 삽입
```

핵심 설계 원칙:
1. **신규 아키텍처를 발명하지 않는다.** AMT를 그대로 사용한다.
2. **신규 학습 목적함수를 발명하지 않는다.** 표준 next-token prediction을 사용한다.
3. **신규 상태머신을 만들지 않는다.** 단순 FIFO 큐를 사용한다.
4. **앰비언트 특화는 토크나이저와 데이터에 한정한다.** 모델 구조나 학습 방식을 건드리지 않는다.

이것이 학문적으로도, 공학적으로도 가장 방어 가능한 접근이다.