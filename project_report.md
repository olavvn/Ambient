# AmbientFlow 프로젝트 종합 리포트 및 발표 자료

이 문서는 `AmbientFlow` 프로젝트의 구현 결과와 향후 계획을 정리한 종합 리포트입니다. 제공해주신 정보를 바탕으로 Contrastive Model과 Transformer Model의 상세 내용, 평가 방법, 향후 연구 및 클라우드 배포 계획을 포함하며, 이를 PPT 발표에 바로 활용할 수 있도록 슬라이드 형태로도 구조화했습니다.

---

## 1. Contrastive Model (테마 추출 모델) 리포트

### 1.1. 사용 목적
앰비언트(Ambient) 음악은 일반적인 팝 음악과 달리 기승전결이 뚜렷하지 않고 모호한 질감과 긴 호흡을 가집니다. 따라서 인간이 수동으로 '테마'를 정의하기 어렵습니다. 주어진 곡에서 음악적 구조를 스스로 파악하고, 곡의 핵심 뼈대가 되는 **테마(Theme) 구간을 자동으로 추출**하기 위해 대조 학습(Contrastive Learning) 기반의 인코더를 도입했습니다.

### 1.2. 모델 구조 및 학습 과정
- **모델 구조:** `SegmentEncoder` (Transformer Encoder 기반) + `NTXentLoss` (Normalized Temperature-scaled Cross Entropy)
- **학습 데이터:** 앰비언트 미디 파일 약 1,000곡
- **학습 설정 (`configs/training_config.yaml`):**
  - `epochs`: 100
  - `batch_size`: 256
  - `learning_rate`: 3.0e-4
  - `temperature`: 0.07 (NT-Xent Loss 용)
  - `segment_len`: 64 토큰

### 1.3. Theme Transformer 논문과의 비교 (구현/미구현)
- **구현된 부분:** 
  - 같은 곡에서 추출한 두 구간(Positive pair)은 임베딩 공간에서 가깝게, 다른 곡에서 추출한 구간(Negative pair)은 멀어지도록 학습하는 **NT-Xent 대조 학습 로직**을 완벽히 구현했습니다.
  - 추출된 임베딩을 바탕으로 밀도 기반 클러스터링(DBSCAN 등)을 수행해 대표 테마를 선정하는 파이프라인을 구축했습니다.
- **구현하지 않은 부분 및 이유:** 
  - 논문에서는 Pitch Shift, Tempo Change 등 복잡한 데이터 증강(Augmentation) 기법을 사용하지만, 앰비언트 음악은 피치나 템포보다 **서스테인(Sustain)과 리버브(Reverb)** 같은 질감이 더 중요하므로 1차원 REMI 토큰 시퀀스의 **슬라이딩 윈도우 분할 방식**으로 증강을 단순화했습니다.
  - **슬라이딩 윈도우 분할 방식 및 이점:** 긴 곡을 통째로 넣는 대신, 고정된 길이(예: 64토큰)의 윈도우(창)를 설정하고 이를 일정 간격(Stride)만큼 옆으로 밀어가며 수많은 조각(Segment) 데이터를 만들어냅니다. 이를 통해 (1) 한정된 수의 미디 파일에서 **수십 배 많은 학습 데이터를 확보**할 수 있고, (2) 모델이 곡의 전체 길이와 무관하게 **'국소적인 패턴(Motif)'을 집중적으로 학습**할 수 있어 테마의 특징을 더 날카롭게 포착하는 이점이 있습니다.

### 1.4. 성능 평가 결과
- 훈련 과정에서 NT-Xent Loss가 안정적으로 수렴함을 확인했습니다. 
- 테스트 미디를 입력했을 때, 앰비언트 음악 내에서 반복적으로 등장하는 핵심 코드 진행과 텍스처 구간이 클러스터링을 통해 대표 테마(Representative Theme)로 성공적으로 군집화 및 추출되었습니다.
  - **추출 결과 확인 방법:** 프로젝트 내 구현된 `test_theme_extractor.py` 스크립트를 실행하거나, Python 환경에서 `ThemeExtractor.extract_from_midi(midi_file)` 함수에 직접 미디 파일을 넣어 반환된 `theme_spans` (시작~종료 초 단위 구간)를 원본 오디오와 비교하여 들으면, 모델이 정확히 어느 구간을 핵심 테마로 지정했는지 직관적으로 귀로 확인할 수 있습니다.

---

## 2. Transformer Model (생성 모델) 리포트

### 2.1. 사용 목적
추출된 테마(뼈대)를 유지하면서도 끝없이 새롭고 자연스러운 변주를 만들어내기 위해 도입했습니다. 단순한 복붙이 아니라, 테마의 분위기를 이해하고 앰비언트 특유의 질감을 가진 새로운 멜로디를 연속 생성합니다.

### 2.2. 모델 구조 및 학습 과정
- **모델 구조:** `AmbientFlowModel` (`ThemeEncoder` + `AmbientMusicDecoder`)
- **핵심 모듈:** 디코더 내부에 **GPA (Gated Parallel Attention)** 메커니즘 탑재
- **학습 데이터:** 테마 정답지가 포함된 앰비언트 미디 청크 약 1,000곡 분량
- **학습 설정 (`configs/training_config.yaml`):**
  - `epochs`: 200
  - `batch_size`: 32
  - `learning_rate`: 1.0e-4
  - `gradient_clip`: 1.0

### 2.3. Theme Transformer 논문과의 비교 (구현/미구현)
- **구현된 부분:** 
  - **Gated Parallel Attention (GPA):** 일반적인 Causal Self-Attention과 테마를 참조하는 Cross-Attention을 학습 가능한 게이트(Gate) 비율로 혼합하여, 모델이 필요할 때만 테마를 참조하도록 하는 핵심 기여를 구현했습니다.
  - **Theme-Aware Cross Entropy:** 테마 참조 토큰(`[THEME_REF]`) 직후의 예측에 더 높은 가중치를 주어 테마 모방 능력을 강화했습니다.
- **구현하지 않은 부분 및 이유:** 
  - 논문은 처음부터 끝까지 정해진 길이의 '한 곡'을 완성하는 구조이지만, 본 프로젝트의 목표는 **'끝없는 실시간 앰비언트 스트리밍'**입니다. 따라서 논문의 구조를 변형하여, 128토큰 단위로 청크(Chunk)를 생성하고 이전 컨텍스트를 슬라이딩 윈도우 방식으로 넘겨주는 **무한 롤링(Rolling) 생성 방식**을 독자적으로 구현했습니다.
  - **무한 롤링 생성 핵심 코드 (`ChunkGenerator` 클래스 내부 구현):**
    ```python
    def _generate_loop(self):
        while self._running:
            # 1. 버퍼에서 현재 테마 가져오기
            theme = self.theme_buffer.get().to(self.device)
            
            with self._context_lock:
                current_context = self.context

            # 2. 모델 추론 (128토큰 청크 단위 생성)
            new_tokens = self.model.generate_chunk(
                theme_tokens=theme,
                context=current_context,
                n_new_tokens=self.chunk_size,
                # ... 샘플링 인자 생략 ...
            )

            # 3. 롤링 컨텍스트: 새로 생성된 토큰을 뒤에 이어 붙이고, 창(Window)을 이동
            with self._context_lock:
                self.context = torch.cat([self.context, new_tokens], dim=1)
                if self.context.size(1) > self.context_len:
                    self.context = self.context[:, -self.context_len:] # 최대 길이 유지

            # 4. 재생 큐(Queue)에 삽입
            self.token_queue.put_nowait_threadsafe(new_tokens[0].tolist())
    ```

### 2.4. 성능 평가 결과
- 앰비언트 음악의 필수 요소인 매우 긴 음표(`DUR_64`)와 서스테인 페달(`SUSTAIN_ON`)을 일관되게 유지하면서 생성합니다. 테마의 핵심 화성은 유지하되 노트의 벨로시티(Velocity)나 리듬을 미세하게 변주하여 단조로움을 탈피하는 생성 결과를 보였습니다.

---

## 3. 모델 성능 평가 시각화 코드 (발표용)
발표 시 Loss 감소 그래프와 Perplexity를 시각화하여 띄울 수 있는 깔끔한 Python(Matplotlib) 코드입니다.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(epochs, loss_values, title="Training Loss"):
    plt.figure(figsize=(10, 5))
    
    # Loss 그래프
    plt.plot(epochs, loss_values, label='Loss', color='#1f77b4', linewidth=2)
    plt.fill_between(epochs, loss_values, alpha=0.2, color='#1f77b4')
    
    # Perplexity (보조축)
    perplexity = np.exp(loss_values)
    ax2 = plt.gca().twinx()
    ax2.plot(epochs, perplexity, label='Perplexity', color='#ff7f0e', linestyle='--', linewidth=2)
    
    # 스타일링
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Cross Entropy)', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    
    # 범례 합치기
    lines_1, labels_1 = plt.gca().get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

# 예시 데이터로 실행 (실제 훈련 중 저장된 loss 배열을 넣으시면 됩니다)
fake_epochs = np.arange(1, 201)
fake_losses = 3.5 * np.exp(-fake_epochs / 40) + 1.2 + np.random.normal(0, 0.05, 200)
plot_training_results(fake_epochs, fake_losses, "Transformer Model: Training Loss & Perplexity")
```

---

## 4. 향후 최적화 및 실험 계획 (Future Works)

### 4.1. 대조 학습 데이터셋 다변화 (Mixed Genre Pre-training)
- **가설:** 현재 앰비언트 음악만으로 대조 학습을 진행하여, 곡의 구조(기승전결)를 명확히 분별하는 능력이 다소 떨어질 수 있습니다.
- **계획:** 팝, 클래식(Maestro), 재즈 등 다양한 장르의 데이터를 섞어 대조 학습(Phase 1)을 진행합니다. 모델이 음악의 보편적인 구조적 경계를 확실히 학습하게 한 뒤, 생성 모델(Phase 3)에서만 앰비언트 데이터로 학습시키면 테마 추출 성능이 비약적으로 상승할 것입니다.

### 4.2. LoRA(Low-Rank Adaptation) 파인튜닝 도입
- **가설:** Transformer 기반 생성 모델은 파라미터가 커질수록 GPU VRAM 소모가 극심합니다.
- **계획:** 거대한 범용 음악 생성 Base 모델을 프리트레이닝 한 후, 앰비언트 데이터셋에 대해서는 **LoRA 기법**을 적용하여 Attention 레이어의 랭크(Rank) 파라미터만 미세 조정(Fine-tuning)합니다. 이를 통해 학습 시간을 단축하고 컴퓨팅 리소스를 극단적으로 절약하면서도 앰비언트 특화 성능을 극대화할 수 있습니다.

### 4.3. 토크나이저 개편
- 리버브(CC91), 서스테인(CC64)에 이어 신디사이저 질감에 필수적인 **Expression(CC11) 및 Modulation(CC1)** 이벤트를 5~8단계로 양자화(Quantization)하여 어휘 사전에 추가하는 실험을 진행합니다.

---

## 5. Streaming Phase 및 클라우드 배포 계획

### 5.1. 연속 스트리밍 (Streaming Phase) 고도화
- **비동기 큐잉:** `ChunkGenerator` 스레드가 128토큰 단위로 선행 생성을 완료하면 `TokenQueue`에 적재하고, 재생 스레드는 이를 가져와 렌더링합니다.
- **실시간 오디오 렌더링 (Ableton Live 연동):** AI 모델이 실시간으로 뱉어내는 텍스트(MIDI 토큰)를 고품질의 사운드로 출력하기 위해, 내부 소프트웨어 신디사이저(`pyfluidsynth`) 뿐만 아니라 **프로페셔널 DAW인 Ableton Live와의 다이렉트 연동**을 구현할 계획입니다. (현재 다른 팀원이 가상 MIDI 포트 통신 및 Ableton 통합 모듈을 개발 중). 이를 통해 모델이 생성한 무한 스트리밍 MIDI 데이터를 Ableton 내부의 최상급 가상악기(VST) 및 오디오 이펙터(고급 앰비언트 리버브, 딜레이 등)로 실시간 전송하여, 상업용 음원 수준의 압도적인 사운드스케이프를 렌더링하게 됩니다.
- **오디오 크로스페이드:** 청크와 청크가 이어질 때 발생하는 '뚝' 끊기는 팝핑(Popping) 노이즈를 제거하기 위해, 오디오 배열 단에서 0.1초 단위의 **Equal-power Crossfade** 연산을 적용하여 완벽하게 연속적인 무한 재생을 구현합니다.

### 5.2. 클라우드 환경 배포 (Cloud Migration)
로컬 GPU 리소스 부족과 실시간 렌더링의 CPU 부하를 해결하기 위한 클라우드 아키텍처 전환 계획입니다.

- **인프라 구성:** AWS EC2 (g4dn.xlarge - NVIDIA T4 GPU) 또는 GCP Compute Engine 활용.
- **아키텍처 분리 (Client-Server):**
  - **Server (Cloud):** FastAPI 기반 추론 서버. 사용자의 시드 미디를 받아 GPU로 텍스트(토큰) 시퀀스만 빠르게 무한 생성합니다.
  - **통신:** WebSocket을 통해 생성된 토큰 스트림을 클라이언트로 실시간 전송합니다.
  - **Client (Local/Web):** 클라이언트는 토큰을 수신받아 Web Audio API나 브라우저 내장 SoundFont(Tone.js 등)를 활용해 로컬 기기에서 소리로 렌더링합니다. 이렇게 하면 클라우드 서버는 오디오 인코딩 부하 없이 오직 텍스트 모델 추론에만 집중할 수 있어 동시 접속자 처리에 매우 유리합니다.

---

## 6. PPT 발표 자료 구조 (Slide Deck Outline)

**[Slide 1] 타이틀**
- 제목: AmbientFlow: 딥러닝 기반 실시간 앰비언트 음악 스트리밍 시스템
- 부제: Theme Transformer 구조를 활용한 테마 추출 및 앰비언트 멜로디 생성

**[Slide 2] 프로젝트 개요 (배경 및 목표)**
- 앰비언트 음악의 특성 (구조 모호, 질감 중심)
- 기존 모델의 한계점 돌파 (테마 자동 추출 + 무한 스트리밍)

**[Slide 3] 전체 시스템 아키텍처**
- 대조 학습(Contrastive) ↔ 생성(Transformer) 이원화 구조 다이어그램
- Tokenizer → Extractor → Chunk Generator 파이프라인

**[Slide 4] Contrastive Model (테마 추출기)**
- 도입 배경: 곡의 구조를 자동으로 군집화하여 핵심 테마 분리
- 아키텍처: Transformer Encoder + NT-Xent Loss
- 구현 특징: 앰비언트에 특화된 슬라이딩 윈도우 기반 데이터 증강

**[Slide 5] Transformer Model (생성기)**
- 아키텍처: GPA (Gated Parallel Attention) 적용
- 구현 특징: 128토큰 단위 Rolling 생성을 통한 무한 스트리밍 달성
- Theme-Aware Cross Entropy를 통한 테마 반영률 극대화

**[Slide 6] 학습 과정 및 데이터셋**
- 사용 데이터: 앰비언트 미디 파일 약 1,000곡
- 학습 파라미터 (Batch size, LR, Epochs 등)

**[Slide 7] 성능 평가 결과**
- (작성한 Python Matplotlib 결과물 시각화 자료 삽입)
- 훈련 손실(Loss) 수렴 차트 및 Perplexity 분석
- 서스테인, 롱노트 등 앰비언트 특성 유지 확인

**[Slide 8] 향후 과제 1: 모델 최적화 및 실험**
- 혼합 장르 대조 학습(Mixed Genre)을 통한 테마 탐지 능력 고도화
- LoRA(Low-Rank Adaptation) 파인튜닝을 통한 컴퓨팅 자원 효율화

**[Slide 9] 향후 과제 2: 스트리밍 및 클라우드 배포**
- Equal-power 크로스페이드로 팝핑 노이즈 제거
- AWS/GCP 기반 클라우드 마이그레이션 전략 (WebSocket 기반 Token Streaming)

**[Slide 10] Q&A 및 마무리**
- 요약 및 질의응답
