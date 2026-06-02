# 앰비언트 음악 특화 토크나이저 설계 보고서

> **목적:** 기존 REMI-Ambient 토크나이저의 문제점 분석 → MiDiTok 기반 최적 토크나이저 설계 근거 제시 → Claude Code 실행 가능한 비교 평가 파이프라인 제안

---

## 1. 기존 토크나이저 버그 및 설계 결함 분석

### 1-1. 치명적 버그 (Critical Bugs) — 사용 불가 수준

#### BUG-01: Duration 인코딩/디코딩 수식 불일치 → 노트 길이 2배 오류

```python
# 인코딩 (tokenizer.py L150)
dur_units = max(1, min(64, round(ev.duration * self.subdivisions / 2)))
# → dur_units = duration * 32 / 2 = duration * 16

# 디코딩 (tokenizer.py L198)
duration = dur_units * (current_beat_dur / (self.subdivisions / self.beats_per_bar))
# bpm=60, subdivisions=32, beats_per_bar=4 대입:
# duration = dur_units * (1.0 / 8) = dur_units / 8
```

**결과:** `decode(encode(d)) = d * 16 / 8 = 2d` → **모든 노트 길이가 2배로 복원됨**

---

#### BUG-02: BAR 토큰이 시간을 누적 전진시킴 → 절대 시간 붕괴

```python
# 디코딩 (tokenizer.py L174)
if tok == BAR_TOKEN:
    current_time += current_beat_dur * self.beats_per_bar  # ← 매번 1마디만큼 전진
```

REMI에서 BAR 토큰은 "새 마디의 시작"을 마킹하는 것이지, 시간을 전진시키는 신호가 아니다. 시간은 POS 토큰으로 정밀 조정되어야 한다. 현재 구현은 BAR마다 4박자씩 무조건 전진하면서, POS 토큰은 디코딩 시 **완전히 무시**한다. 결과적으로 노트 시작 시간이 전부 틀어진다.

---

#### BUG-03: `_time_to_bar` / `_time_to_pos`에 BPM 하드코딩

```python
def _time_to_bar(self, time: float, bpm: float = 60.0) -> int:  # bpm 항상 60
def _time_to_pos(self, time: float, bpm: float = 60.0) -> int:  # bpm 항상 60
```

MIDI 내부 템포 변화가 있어도 전혀 반영되지 않는다. 앰비언트 음악은 70 BPM 이하가 많은데, 60 BPM로 고정하면 모든 bar/position 경계가 어긋난다.

---

### 1-2. 앰비언트 음악과 구조적으로 맞지 않는 설계

#### DESIGN-01: 최대 Duration 표현 범위 부족

```python
DUR_TOKENS = [f"DUR_{d}" for d in list(range(1, 33)) + [48, 64]]
# 최대 DUR_64
# 60 BPM, subdivisions=32 기준: 1 unit ≈ 0.0625박자
# DUR_64 = 4박자 = 1마디
```

앰비언트 패드는 흔히 **4~8마디 이상** 지속된다. DUR_64가 1마디라면, 4마디 지속 패드는 `DUR_64`로 클리핑되어 모두 1마디로 뭉개진다. 앰비언트에서 가장 중요한 정보가 손실된다.

---

#### DESIGN-02: 템포 범위가 앰비언트 커버 불가

```python
TEMPO_TOKENS = [f"TEMPO_{t}" for t in range(40, 125, 5)]
# 범위: 40~120 BPM
```

앰비언트 음악의 일반적 BPM 분포:
- Brian Eno 계열: **20~50 BPM**
- Stars of the Lid 계열: **30~60 BPM**
- Dark ambient: **20~40 BPM**

40 BPM 미만은 클리핑되어 모두 TEMPO_40으로 처리되고, 인코딩 시 `max(40, ...)` 강제로 정보가 손실된다.

---

#### DESIGN-03: 벨로시티 양자화 8단계 — 앰비언트 다이나믹 표현 불가

```python
VEL_TOKENS = [f"VEL_{v}" for v in range(8, 120, 14)]
# → [8, 22, 36, 50, 64, 78, 92, 106] — 8단계
```

앰비언트 음악의 핵심은 **pp ~ mp 범위(1~60)의 미세한 다이나믹 변화**다. 8단계 양자화에서 이 범위는 겨우 5단계(8, 22, 36, 50, 64)로 표현되며, 단계 간격이 14로 너무 크다.

---

#### DESIGN-04: CC 표현이 앰비언트에 필수적인 컨트롤러를 누락

현재 코드는 CC64(서스테인)와 CC91(리버브)만 처리한다. 앰비언트 신디사이저 제어에 실제 사용되는 CC:

| CC 번호 | 이름 | 앰비언트 활용 | 현재 지원 |
|---|---|---|---|
| CC1 | Modulation | 패드 음색 변조, 비브라토 | ❌ |
| CC7 | Channel Volume | 페이드 인/아웃 | ❌ |
| CC10 | Pan | 공간감 이동 | ❌ |
| CC11 | Expression | 실시간 음량 표현 | ❌ |
| CC64 | Sustain Pedal | 음 유지 | ✅ |
| CC71 | Timbre/Resonance | 필터 공명 | ❌ |
| CC74 | Brightness | 로우패스 필터 컷오프 | ❌ |
| CC91 | Reverb Depth | 공간감 깊이 | ✅ (5단계만) |
| CC93 | Chorus Depth | 코러스 효과 | ❌ |

---

#### DESIGN-05: REVERB 5단계 양자화 — 앰비언트의 핵심 정보 손실

```python
REVERB_TOKENS = ["REVERB_0", "REVERB_32", "REVERB_64", "REVERB_96", "REVERB_127"]
```

앰비언트에서 리버브 깊이는 **음악적 의미를 가진 표현 수단**이다. 32 단위 간격의 5단계는 공간감의 미세한 변화를 전혀 포착하지 못한다. 최소 16단계(8 간격) 이상이 필요하다.

---

#### DESIGN-06: BEAT_TOKENS 정의되었으나 인코딩/디코딩 모두 미사용

```python
BEAT_TOKENS = [f"BEAT_{i}" for i in range(1, 5)]  # 어휘에 포함
# _events_to_tokens 내 어디에도 emit되지 않음
# tokens_to_midi 내 어디에도 처리 코드 없음
```

Dead code이지만 어휘 크기에서 4개 슬롯을 낭비한다.

---

#### DESIGN-07: 테마 특수 토큰이 앰비언트 전용 시스템에서 불필요

```python
SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[MASK]",
                  "[THEME_S]", "[THEME_E]", "[THEME_REF]", "[SEP]"]
```

앞선 논의에서 결론이 난 대로, 앰비언트에서 Theme Transformer 방식의 테마 마킹은 불필요하다. 어휘 슬롯 3개 낭비.

---

#### DESIGN-08: Bar 기반 시간 추적이 앰비언트에 구조적으로 부적합

REMI의 Bar+Position 시스템은 **명확한 박자 그리드가 있는 음악**을 전제한다. 앰비언트에서:
- 노트가 마디 경계를 무시하고 자유롭게 시작/종료
- 같은 BPM 내에서도 루바토(rubato)적 흐름
- 마디라는 개념 자체가 희박

Bar 토큰이 의미 있는 구조적 단위가 아닌데 매 마디마다 삽입되면, 트랜스포머가 학습해야 할 의미 없는 노이즈가 추가된다.

---

### 1-3. 문제 요약표

| 코드 위치 | 분류 | 심각도 | 설명 |
|---|---|---|---|
| L150, L198 | BUG-01 | 🔴 치명적 | Duration 2배 오류 |
| L174 | BUG-02 | 🔴 치명적 | BAR 토큰이 시간 누적 전진 |
| L215, L219 | BUG-03 | 🔴 치명적 | BPM 60 하드코딩 |
| L20 | DESIGN-01 | 🟠 높음 | Duration 최대 1마디 제한 |
| L18 | DESIGN-02 | 🟠 높음 | 템포 최소 40 BPM 제한 |
| L21 | DESIGN-03 | 🟡 중간 | 벨로시티 8단계만 |
| L103~106 | DESIGN-04 | 🟠 높음 | CC 2개만 지원 |
| L23 | DESIGN-05 | 🟡 중간 | 리버브 5단계만 |
| L16 | DESIGN-06 | 🔵 낮음 | BEAT 토큰 미사용 |
| L12~13 | DESIGN-07 | 🔵 낮음 | 테마 토큰 불필요 |
| 전체 구조 | DESIGN-08 | 🟠 높음 | Bar 기반 시간 추적 |

---

## 2. 앰비언트 음악 특성 기반 토크나이저 요구사항 정의

앰비언트 음악의 물리적·음악적 특성에서 요구사항을 도출한다.

| 특성 | 요구사항 |
|---|---|
| 긴 노트 지속 (4~32마디) | Duration 토큰 범위 대폭 확장 |
| 느린 템포 (20~70 BPM) | Tempo 토큰 하한 20 BPM으로 낮춤 |
| 미세한 다이나믹 (pp~mp) | 벨로시티 16단계 이상 |
| 헤비 리버브 (공간 표현의 핵심) | 리버브 16단계 이상 |
| 필터 스윕 (CC74) | 브라이트니스 CC 추가 |
| 볼륨 오토메이션 (CC7, CC11) | 익스프레션 CC 추가 |
| 비박자적 시간 흐름 | TimeShift 기반 시간 추적 |
| 희소한 음표 밀도 | 긴 Silence 처리 가능한 방식 |

---

## 3. MiDiTok 토크나이저 비교 분석

### 3-1. 후보 토크나이저별 앰비언트 적합성 평가

#### REMI (기존 방식)

| 항목 | 평가 |
|---|---|
| 시간 표현 | Bar + Position (마디 단위 이산적) |
| 장음 처리 | Duration 토큰으로 표현 가능하나 범위 설정 필요 |
| 긴 침묵 처리 | BAR 토큰으로 자동 스킵 가능 |
| 앰비언트 적합성 | **낮음** — 마디 구조가 앰비언트의 유동적 시간과 충돌 |

#### MIDI-Like

| 항목 | 평가 |
|---|---|
| 시간 표현 | TimeShift (연속적) + NoteOn/NoteOff |
| 장음 처리 | NoteOff 기반 — 겹친 음표 처리 시 오프셋 왜곡 |
| 긴 침묵 처리 | **문제 있음** — 최대 TimeShift 값으로 클리핑 |
| 앰비언트 적합성 | **낮음** — 긴 침묵과 긴 음표 모두 NoteOff로 처리 불안정 |

#### TSD (Time Shift Duration)

| 항목 | 평가 |
|---|---|
| 시간 표현 | TimeShift (연속적, 절대 시간 기반) |
| 장음 처리 | 명시적 Duration 토큰 — NoteOff보다 안정적 |
| 긴 침묵 처리 | 최대 TimeShift 설정값까지 표현 가능 |
| 앰비언트 적합성 | **중간** — TimeShift 범위만 충분히 크면 적합 |

#### Structured

| 항목 | 평가 |
|---|---|
| 시간 표현 | TimeShift (고정 패턴: Pitch→Vel→Dur→TimeShift) |
| 장음 처리 | Duration 명시적 |
| 긴 침묵 처리 | 최대 TimeShift로 클리핑 |
| 앰비언트 적합성 | **낮음** — 고정 패턴이 CC 이벤트 삽입 불가 |

#### CPWord (Compound Word)

| 항목 | 평가 |
|---|---|
| 시간 표현 | Bar + Position (REMI 기반) |
| 장음 처리 | Duration 포함 compound |
| 시퀀스 길이 | 대폭 단축 (pooling) |
| 앰비언트 적합성 | **낮음** — 소형 모델 비권장, 마디 기반 |

#### Octuple

| 항목 | 평가 |
|---|---|
| 시간 표현 | Bar + Position |
| 시퀀스 길이 | 매우 짧음 (음표당 1 pooled token) |
| 앰비언트 적합성 | **낮음** — 소형 모델 비권장, CC 표현 어려움 |

---

### 3-2. 최종 권장: **TSD 기반 커스텀 앰비언트 토크나이저**

#### 선택 근거

**1. TimeShift 기반 시간 표현이 앰비언트에 구조적으로 맞다**

REMI의 Bar+Position 방식은 마디 경계라는 이산적 그리드를 전제한다. TSD의 TimeShift는 이전 이벤트로부터의 **상대적 시간 간격**을 표현하므로, 마디 개념 없이 자유롭게 시간을 표현할 수 있다. 앰비언트의 유동적 시간 흐름과 일치한다.

**2. 명시적 Duration 토큰이 긴 패드 음표에 안정적이다**

MIDI-Like의 NoteOff 방식은 겹치는 음표 처리 시 오프셋이 왜곡되는 문제가 있다. 앰비언트는 다성부 패드가 겹치는 상황이 빈번하므로, Duration을 직접 토큰으로 명시하는 TSD가 훨씬 안정적이다.

**3. 트랜스포머의 어텐션 패턴과 TimeShift의 궁합**

TimeShift는 이벤트 간 시간 간격을 명시적으로 나타내므로, 트랜스포머가 어텐션을 통해 "얼마나 먼 과거의 이벤트인가"를 직접 학습할 수 있다. Bar 기반은 마디 내 포지션이라는 간접적 표현을 사용하므로 장기 패턴 학습에 불리하다.

---

## 4. 앰비언트 특화 TSD 토크나이저 설계안

### 4-1. 어휘 설계

```python
from miditok import TSD, TokenizerConfig

# 앰비언트 특화 설정
AMBIENT_CONFIG = TokenizerConfig(
    # ── 피치 범위: 전체 MIDI (신디사이저 특성 반영)
    pitch_range=(21, 108),  # A0~C8

    # ── 시간 해상도: 박자당 8분할 (앰비언트는 정밀 그리드 불필요)
    # 앰비언트는 정확한 온셋보다 대략적 타이밍이 중요
    beat_res={(0, 8): 8, (8, 64): 4},
    # → 0~8박: 8분할 / 8박 이상(긴 음표/침묵): 4분할

    # ── 추가 토큰
    use_tempos=True,
    use_velocities=True,
    use_programs=False,  # 단일 악기 (앰비언트 패드)
    use_chords=False,    # 코드 분석 불필요
    use_rests=True,      # 침묵 명시 (앰비언트에서 중요)
    use_time_signatures=False,  # 박자 고정 필요 없음

    # ── 템포: 20~80 BPM (앰비언트 실제 범위)
    tempo_range=(20, 80),
    nb_tempos=30,  # 30단계

    # ── 벨로시티: 16단계 (pp~mf 미세 표현)
    nb_velocities=16,

    # ── 특수 토큰
    special_tokens=["PAD", "BOS", "EOS", "MASK"],

    # ── TSD 확장 Duration (핵심)
    # beat_res의 최대값 기준으로 Duration 토큰 자동 생성됨
    # 추가로 additional_params에서 최대 Duration 지정
    additional_params={
        "max_duration": (32, 0, 8),
        # → 32박자 * 8분할 = 최대 256유닛
        # 60 BPM 기준: 32박자 = 32초 (8마디)
    }
)
```

### 4-2. CC 이벤트 처리를 위한 커스텀 확장

MiDiTok의 TSD를 상속하여 앰비언트 핵심 CC를 처리하는 커스텀 토크나이저를 구축한다.

```python
AMBIENT_CC_VOCAB = {
    # CC번호: (토큰명 prefix, 단계 수, [대표값 리스트])
    7:  ("VOL",     16, list(range(0, 128, 8))),   # Channel Volume
    11: ("EXPR",    16, list(range(0, 128, 8))),   # Expression
    74: ("BRIGHT",  16, list(range(0, 128, 8))),   # Filter Cutoff (앰비언트 핵심)
    91: ("REVERB",  16, list(range(0, 128, 8))),   # Reverb Depth (앰비언트 핵심)
    93: ("CHORUS",   8, [0, 18, 36, 54, 72, 90, 109, 127]),  # Chorus
}
```

**CC 설계 근거:**

- **CC74 (Brightness/Filter Cutoff)**: 앰비언트 패드의 필터 스윕은 텍스처 변화의 핵심 수단. 16단계로 충분한 해상도 확보.
- **CC91 (Reverb)**: 공간감 변화. 기존 5단계에서 16단계로 확장.
- **CC7/CC11 (Volume/Expression)**: 앰비언트의 페이드 인/아웃을 표현하기 위한 필수 CC.

### 4-3. 최종 어휘 크기 추정

| 토큰 그룹 | 토큰 수 | 비고 |
|---|---|---|
| 특수 토큰 | 4 | PAD, BOS, EOS, MASK |
| TimeShift | ~72 | beat_res 설정 기반 자동 |
| Pitch | 88 | A0~C8 |
| Duration | ~80 | max_duration=(32,0,8) |
| Velocity | 16 | 16단계 |
| Tempo | 30 | 20~80 BPM |
| Rest | ~20 | 침묵 표현 |
| CC7 (Volume) | 16 | |
| CC11 (Expression) | 16 | |
| CC74 (Brightness) | 16 | |
| CC91 (Reverb) | 16 | |
| CC93 (Chorus) | 8 | |
| **총계** | **~382** | |

기존 토크나이저 (ALL_TOKENS 기준: 약 246개) 대비 소폭 증가하나, 표현력은 대폭 향상된다.

---

## 5. 토크나이저 객관적 비교 평가 방법론

### 5-1. 평가 지표 정의

#### Metric 1: Reconstruction Fidelity (복원 충실도)

MIDI → 토큰 → MIDI 의 왕복 변환 후 원본과의 차이를 측정한다.

```
RF_pitch    = 복원된 음표 중 피치가 일치하는 비율
RF_onset    = 복원된 음표 중 onset 오차 < threshold인 비율  (threshold=50ms)
RF_duration = 복원된 음표 중 duration 오차 < threshold인 비율 (threshold=100ms)
RF_total    = F1(RF_pitch, RF_onset, RF_duration)의 조화평균
```

**앰비언트 특화 가중치:** Duration 오차에 2배 가중치. 앰비언트에서 음표 길이가 가장 중요한 정보이므로.

---

#### Metric 2: Sequence Length Efficiency (시퀀스 길이 효율)

같은 음악을 표현하는 데 필요한 토큰 수.

```
SLE = tokens_per_second = total_tokens / music_duration_seconds
```

트랜스포머의 컨텍스트 길이는 제한되어 있으므로, 동일한 음악을 더 적은 토큰으로 표현할수록 더 긴 시간적 문맥을 학습할 수 있다.

---

#### Metric 3: CC Preservation Rate (CC 보존율)

```
CPR = |CC events successfully round-tripped| / |total CC events in original|
```

앰비언트에서 CC는 음악적 표현의 핵심이므로 별도 지표로 측정.

---

#### Metric 4: Duration Quantization Error (지속 시간 양자화 오차)

```
DQE_mean = mean(|original_duration - quantized_duration| / original_duration)
DQE_long = mean for notes with duration > 2.0 seconds  (앰비언트 특화)
```

특히 긴 노트(2초 이상)에서의 양자화 오차를 별도 측정. 앰비언트에서 긴 노트의 Duration 정밀도가 핵심.

---

#### Metric 5: Tempo Range Coverage (템포 범위 커버리지)

```
TRC = |MIDI tempos successfully represented without clipping| / |total tempo events|
```

---

#### Metric 6: Vocabulary Utilization (어휘 활용률)

```
VU = |unique tokens used in dataset| / |total vocabulary size|
```

어휘는 크지만 실제로 사용되는 토큰이 적으면 비효율적 설계다.

---

### 5-2. Claude Code 실행 파이프라인

아래 지시사항을 Claude Code에 그대로 입력하여 비교 평가를 실행할 수 있다.

---

#### STEP 0: 환경 설정

```bash
pip install miditok pretty_midi numpy pandas matplotlib seaborn mir_eval
```

---

#### STEP 1: 테스트 데이터셋 준비

```
다음 조건을 만족하는 MIDI 파일 20개를 준비하거나 생성해줘:
- 앰비언트 스타일 (긴 노트, 느린 템포)
- 노트 지속 시간 분포: 1초 미만 ~ 30초 이상 포함
- 템포: 30~70 BPM
- CC91 (리버브), CC74 (브라이트니스) 포함
- 총 길이: 각 30초~3분

또는 테스트용 합성 MIDI를 생성하는 코드를 작성해줘:
```

**합성 MIDI 생성 코드 (Claude Code에 요청):**

```
pretty_midi를 사용해서 앰비언트 스타일 테스트 MIDI 20개를 생성하는 
generate_ambient_test_midi.py를 작성해줘.

요구사항:
- 각 파일: 60~180초
- 템포: 25~65 BPM 균등 샘플링
- 음표 지속 시간: log-uniform 분포로 0.5초~30초
- CC91 리버브: 40~127 범위에서 시간에 따라 변화
- CC74 브라이트니스: 0~100 범위에서 변화
- 화음: C장조, F장조, G장조, Am 위주
- 파일 저장: ./test_midi/ 폴더
```

---

#### STEP 2: 토크나이저 구현 파일 작성

**Claude Code에 요청할 내용:**

```
다음 3개의 토크나이저를 miditok 기반으로 구현한 파일을 작성해줘.

[토크나이저 1] tokenizer_remi_original.py
- 기존 REMIAmbientTokenizer 그대로 (버그 포함, 베이스라인 측정용)
- 파일 경로: ./tokenizers/remi_original.py

[토크나이저 2] tokenizer_remi_fixed.py  
- miditok.REMI 기반
- 설정:
  - beat_res = {(0, 4): 8, (4, 12): 4}
  - tempo_range = (20, 80)
  - nb_velocities = 16
  - use_rests = True
  - max_bar_embedding = 256
- Duration 버그 수정 버전
- 파일 경로: ./tokenizers/remi_fixed.py

[토크나이저 3] tokenizer_tsd_ambient.py
- miditok.TSD 기반 + CC 확장 커스텀 클래스
- 설정:
  - beat_res = {(0, 8): 8, (8, 64): 4}  
  - tempo_range = (20, 80)
  - nb_velocities = 16
  - use_rests = True
  - additional_params["max_duration"] = (32, 0, 8)
- CC 처리: CC7, CC11, CC74, CC91, CC93를 별도 토큰으로 인코딩
- miditok.TSD를 상속하고 _create_track_events()를 오버라이드하여 CC 이벤트 삽입
- 파일 경로: ./tokenizers/tsd_ambient.py
```

---

#### STEP 3: 비교 평가 스크립트 작성

**Claude Code에 요청할 내용:**

```
다음 스펙으로 토크나이저 비교 평가 스크립트 evaluate_tokenizers.py를 작성해줘.

입력:
- ./test_midi/ 폴더의 MIDI 파일들
- ./tokenizers/ 폴더의 토크나이저 3개

출력:
- ./results/metrics_summary.csv  (토크나이저별 지표 요약)
- ./results/per_file_metrics.csv (파일별 상세 지표)
- ./results/plots/ (시각화)

측정할 지표 (각 MIDI 파일에 대해 각 토크나이저 적용):

1. Reconstruction Fidelity (RF)
   - MIDI → 토큰 → MIDI 왕복 변환 수행
   - mir_eval.transcription.precision_recall_f1_overlap 사용
   - onset tolerance: 50ms, offset tolerance: 50ms
   - pitch 일치 여부 포함
   - RF_pitch, RF_onset, RF_duration, RF_f1 계산

2. Sequence Length Efficiency (SLE)  
   - tokens_per_second = len(tokens) / midi.get_end_time()
   - long_note_token_overhead = tokens for notes > 2sec / total note tokens
   
3. CC Preservation Rate (CPR)
   - CC91, CC74에 대해 원본 CC 이벤트 수 vs 복원된 CC 이벤트 수
   - 타이밍 오차 허용: 200ms
   
4. Duration Quantization Error (DQE)
   - 전체 노트: mean absolute relative error
   - 긴 노트 (>2초): 별도 측정
   - 히스토그램 데이터 저장 (duration_bin별 오차)

5. Tempo Range Coverage (TRC)
   - 원본 템포 이벤트 중 클리핑 없이 표현된 비율

6. Vocabulary Utilization (VU)
   - 데이터셋 전체에서 실제 사용된 고유 토큰 수 / 전체 어휘 크기

시각화 요구사항 (matplotlib/seaborn):
- Figure 1: RF 지표 radar chart (3 토크나이저 비교)
- Figure 2: Duration 오차 분포 boxplot (duration 구간별: <1s, 1-5s, 5-15s, >15s)
- Figure 3: Sequence length efficiency bar chart
- Figure 4: CC preservation heatmap (토크나이저 × CC 번호)
- Figure 5: Vocabulary utilization pie chart (각 토크나이저)
```

---

#### STEP 4: 결과 해석 보고서 자동 생성

**Claude Code에 요청할 내용:**

```
evaluate_tokenizers.py 실행 후 생성된 ./results/metrics_summary.csv를 읽어서
다음 형식의 결과 해석 보고서 result_interpretation.md를 생성해줘.

포함 내용:
1. 각 지표별 최고 성능 토크나이저 및 수치
2. 앰비언트 음악에 가장 중요한 지표(Duration, CC)에서의 순위
3. 시퀀스 효율 vs 복원 충실도 트레이드오프 분석
4. 최종 권장 토크나이저 선정 및 이유
5. 선정된 토크나이저의 개선 가능한 하이퍼파라미터 제안
```

---

### 5-3. 예상 결과 해석 기준표

| 지표 | REMI Original | REMI Fixed | TSD Ambient | 중요도 |
|---|---|---|---|---|
| RF_f1 | 매우 낮음 (버그) | 중간 | 높음 예상 | ★★★ |
| DQE (>2초 노트) | 매우 높음 | 중간 | 낮음 예상 | ★★★★★ |
| CPR (CC91) | 낮음 (5단계) | 낮음 (5단계) | 높음 (16단계) | ★★★★ |
| SLE | 중간 | 중간 | 낮음 예상 | ★★ |
| TRC | 낮음 (40 BPM 하한) | 높음 | 높음 | ★★★ |
| VU | 낮음 (Dead tokens) | 중간 | 높음 예상 | ★★ |

> **판정 기준:** DQE와 CPR에서 TSD Ambient가 우세하면 채택. SLE가 다소 불리하더라도 컨텍스트 길이 제한은 하이퍼파라미터로 대응 가능.

---

## 6. 추가 고려사항: BPE (Byte Pair Encoding) 적용 가능성

MiDiTok은 토크나이저 학습 후 BPE 적용을 지원한다.

```python
# BPE 적용으로 시퀀스 길이 단축 가능
tokenizer.train(
    vocab_size=1000,  # 기본 어휘 ~382 → BPE로 1000까지 확장
    model="BPE",
    files_paths=midi_files
)
```

앰비언트 음악에서 자주 등장하는 패턴 (예: `TimeShift_2.0 + PITCH_60 + VEL_32 + DUR_16`)이 단일 BPE 토큰으로 압축되면, 시퀀스 길이가 30~50% 단축된다. 이는 트랜스포머의 컨텍스트 한계 내에서 더 긴 음악 구조를 학습할 수 있게 한다.

**Claude Code에서 BPE 비교 추가 요청:**

```
evaluate_tokenizers.py에 TSD Ambient + BPE(vocab_size=800) 변형을 
4번째 토크나이저로 추가해줘.
BPE 학습은 test_midi 데이터셋 전체로 수행.
BPE 적용 전후 SLE 변화량을 별도 지표로 추가.
```

---

## 7. 전체 작업 흐름 요약

```
[분석 단계]
기존 tokenizer.py 버그 확인 → 이 보고서의 분석 내용 공유

[설계 단계]  
Claude Code: generate_ambient_test_midi.py 작성 및 실행
Claude Code: tokenizers/ 폴더에 3개 토크나이저 구현

[평가 단계]
Claude Code: evaluate_tokenizers.py 작성 및 실행
Claude Code: result_interpretation.md 자동 생성

[선정 단계]
DQE + CPR + TRC 기준으로 최적 토크나이저 확정
선정된 토크나이저로 앰비언트 MIDI 데이터셋 전체 전처리
```

---

## 참고: 앰비언트 MIDI에서 중요한 CC 번호 전체 목록

전사(transcription) 도구가 어떤 CC를 보존하는지 확인 후, 토크나이저 어휘에 반영할 것.

| CC | 이름 | 앰비언트 역할 | 우선순위 |
|---|---|---|---|
| 1 | Modulation | 비브라토, 표현 | 중 |
| 7 | Channel Volume | 전체 볼륨 | 높음 |
| 10 | Pan | 좌우 공간 | 중 |
| 11 | Expression | 세밀한 음량 | 높음 |
| 64 | Sustain | 음 유지 | 높음 |
| 71 | Timbre/Resonance | 필터 공명 | 중 |
| 74 | Brightness | 필터 컷오프 | **최고** |
| 91 | Reverb Depth | 공간감 | **최고** |
| 93 | Chorus | 코러스 | 낮음 |

---

*작성 기준: MiDiTok v3.x 문서 및 앰비언트 음악 물리적 특성 분석*
*Claude Code 실행 환경: Python 3.10+, CUDA 선택적*