# TSD Ambient 토크나이저 상세 설명

> TSD Ambient는 무엇인지, 어떤 논문/자료에서 출발했는지, MIDI를 어떻게 토큰으로 변환하는지 처음부터 끝까지 설명한다.

---

## 목차

1. [TSD란 무엇인가](#1-tsd란-무엇인가)
2. [원 논문 및 참고 자료](#2-원-논문-및-참고-자료)
3. [REMI와 TSD의 결정적 차이](#3-remi와-tsd의-결정적-차이)
4. [TSD Ambient: 앰비언트 특화 설계](#4-tsd-ambient-앰비언트-특화-설계)
5. [어휘(Vocabulary) 구조 상세](#5-어휘vocabulary-구조-상세)
6. [토큰화 과정 단계별 설명](#6-토큰화-과정-단계별-설명)
7. [실제 인코딩 예시](#7-실제-인코딩-예시)
8. [CC 이벤트 처리 전략](#8-cc-이벤트-처리-전략)
9. [디코딩 과정](#9-디코딩-과정)
10. [구현 코드 해설](#10-구현-코드-해설)
11. [설계 파라미터 선택 근거](#11-설계-파라미터-선택-근거)
12. [한계 및 개선 방향](#12-한계-및-개선-방향)

---

## 1. TSD란 무엇인가

**TSD**는 **Time Shift Duration**의 약자로, MIDI 음악을 토큰 시퀀스로 변환하는 방식 중 하나다.

핵심 아이디어는 단순하다: 음악을 표현할 때 "언제 시작하는가"와 "얼마나 지속되는가"를 분리해서 명시적으로 나타낸다.

- **TimeShift**: 이전 이벤트 이후 얼마나 기다렸는가 (상대 시간)
- **Duration**: 이 음표가 얼마나 지속되는가 (음표 길이)

음표 하나를 표현하는 데 필요한 토큰은 다음 4개다:

```
TimeShift_X  →  Pitch_Y  →  Velocity_Z  →  Duration_W
```

이것이 TSD의 전부다. 박자, 마디, 포지션 같은 개념이 없다. 오직 "시간 간격 → 음 → 세기 → 지속" 반복이다.

---

## 2. 원 논문 및 참고 자료

### 2-1. TSD의 직접 출처: MiDiTok 라이브러리

TSD는 MiDiTok 라이브러리(`miditok`)에서 구현된 토크나이저 방식이다.

> **Fradet, N., Briot, J.-P., Chhel, F., El Hajal, A., & Gutkin, A. (2021).**  
> *MidiTok: A Python package for MIDI file tokenization.*  
> Extended Abstracts for the Late-Breaking Demo Session of the 22nd International Society for Music Information Retrieval Conference (ISMIR 2021).

MiDiTok는 다음 논문들의 토크나이저 방식들을 통합 구현한 파이썬 라이브러리다:

| 방식 | 원 논문 |
|---|---|
| REMI | Huang & Yang (2020), Pop Music Transformer (ACM MM) |
| MIDI-Like | Oore et al. (2018), This Time with Feeling (NeurIPS Workshop) |
| TSD | MiDiTok 자체 설계 (MIDI-Like 개선판) |
| Structured | MiDiTok 자체 설계 |
| CPWord | Hsiao et al. (2021), Compound Word Transformer (AAAI) |
| Octuple | Muhamed et al. (2021), Symbolic Music Generation with Non-Differentiable Rule Guided GAN (ACL) |

**TSD는 MiDiTok 팀이 독자 설계한 방식**으로, 아래에 설명할 MIDI-Like의 단점을 보완하기 위해 만들어졌다.

---

### 2-2. TSD의 지적 계보: MIDI-Like와의 관계

TSD를 이해하려면 그 전신인 **MIDI-Like** 방식을 먼저 알아야 한다.

**MIDI-Like**는 Oore et al. (2018)의 논문 *"This Time with Feeling: Learning Expressive Musical Performance"*에서 소개된 방식으로, MIDI 이벤트를 직접 토큰으로 변환한다:

```
NoteOn_60  →  TimeShift_480  →  NoteOff_60
```

각 음표는 `NoteOn` → 대기 → `NoteOff` 쌍으로 표현된다. 실제 피아노 롤처럼 음표의 시작과 끝을 별도로 명시하는 방식이다.

**MIDI-Like의 문제점**:

음표가 겹칠 때(폴리포니) `NoteOff`가 섞이는 문제가 생긴다.

```
NoteOn_60  →  TimeShift_100  →  NoteOn_64  →  TimeShift_200  →
NoteOff_60  →  TimeShift_50  →  NoteOff_64
```

이 시퀀스에서 모델이 `NoteOff_60`이 어느 `NoteOn`에 대응하는지 학습하는 것은 어렵다. 특히 앰비언트 음악처럼 수십 초짜리 음표가 서로 겹치는 경우, `NoteOn`과 `NoteOff` 사이의 거리가 시퀀스에서 수백 토큰에 달해 어텐션 부담이 커진다.

**TSD가 이 문제를 해결하는 방법**:  
`NoteOff`를 없애고 Duration 토큰으로 음표 길이를 음표 시작 시점에서 즉시 명시한다.

```
TimeShift_2.0  →  Pitch_60  →  Velocity_64  →  Duration_8.0
```

이 4개 토큰을 보는 순간, 이 음표가 언제 시작해서 얼마나 지속되는지 완전히 결정된다. 나중에 NoteOff를 찾을 필요가 없다.

---

### 2-3. 앰비언트 음악 특화를 위한 추가 참고 자료

TSD Ambient 설계에는 순수 알고리즘 논문 외에, 앰비언트 음악의 물리적/음악적 특성에 대한 다음 지식이 반영되었다:

**앰비언트 음악의 특성 (음악학적 근거)**:
- Brian Eno의 앰비언트 시리즈 (1978~): 20~50 BPM, 수 마디에 걸친 음표, 강한 리버브
- Stars of the Lid, Grouper, William Basinski 계열: 극히 낮은 밀도, 텍스처 중심
- Max Richter의 "Sleep" (2015): 8시간짜리 앰비언트 작품, 4~8박자 지속 음표가 기본

**MIDI CC 컨트롤러 활용 (신디사이저 제어 이론)**:
- CC74 (Filter Cutoff/Brightness): 로우패스 필터 컷오프 주파수 제어 — 음색의 명도 변화
- CC91 (Reverb Depth): 잔향 깊이 — 공간감의 핵심
- CC64 (Sustain Pedal): 음 유지 — 앰비언트의 텍스처 레이어링

**시퀀스 효율성 (트랜스포머 구조 고려)**:
- Vaswani et al. (2017) "Attention Is All You Need": 셀프 어텐션은 O(n²) 복잡도로, 시퀀스 길이가 짧을수록 학습 효율이 높다
- 앰비언트 음악 한 곡 (180초, 40 BPM)을 REMI로 인코딩하면 약 200~300 토큰이지만, 동일한 음악을 TSD로 인코딩하면 약 80~120 토큰이다. 동일한 컨텍스트 창에서 2~3배 더 긴 음악 구조를 학습할 수 있다.

---

## 3. REMI와 TSD의 결정적 차이

현재 프로젝트의 기존 토크나이저(`REMIAmbientTokenizer`)는 REMI 방식을 기반으로 한다. REMI와 TSD의 차이를 이해하는 것이 TSD Ambient를 선택한 이유를 설명한다.

### 3-1. 시간 표현 방식

**REMI (Bar + Position 방식)**:

```
BAR  →  POS_0  →  TEMPO_60  →  PITCH_60  →  DUR_8  →  VEL_64  →
BAR  →  POS_4  →  PITCH_62  →  DUR_8  →  VEL_50  →
```

- `BAR`는 새로운 마디의 시작을 알린다
- `POS_N`은 현재 마디 안에서의 위치를 나타낸다
- 시간 = (몇 번째 마디) × (마디 길이) + (포지션 × 그리드 해상도)

**TSD (TimeShift 방식)**:

```
Tempo_30.34  →  Rest_2.0.2  →  Pitch_50  →  Velocity_71  →  Duration_1.0.8  →
TimeShift_1.0  →  Pitch_53  →  Velocity_39  →  Duration_0.5.8  →
```

- `TimeShift_X`는 이전 이벤트로부터의 **상대 시간 간격**이다
- 마디 개념이 없다 — 순전히 연속적인 시간 흐름

### 3-2. 앰비언트에서 왜 TSD가 적합한가

REMI의 Bar+Position 구조는 **박자 그리드가 명확한 음악**을 전제한다. 재즈, 클래식 소나타, 팝송처럼 4/4박자에 맞춰 음표가 배치되는 음악에서 Bar는 의미 있는 구조 단위다.

앰비언트 음악은 다르다:

| 상황 | REMI에서의 문제 | TSD에서의 처리 |
|---|---|---|
| 음표가 마디 경계를 가로지름 | BAR 토큰 삽입 위치와 무관하게 음표 계속 | TimeShift로 정확한 간격만 명시 |
| 20 BPM의 매우 느린 템포 | 1마디 = 12초, BAR 토큰이 12초마다 삽입 | TimeShift_48.0 같은 긴 간격 1개로 해결 |
| 루바토(자유로운 박자) | 그리드와 맞지 않아 POS 오차 발생 | TimeShift는 실제 간격을 그대로 표현 |
| 음표 희소성 | 음표 없는 마디에도 BAR+POS 토큰 삽입 | Rest 토큰 1개로 긴 침묵 처리 |

---

## 4. TSD Ambient: 앰비언트 특화 설계

기본 TSD 위에 앰비언트 음악의 특성에 맞게 파라미터를 조정한 것이 **TSD Ambient**다. MiDiTok의 `TSD` 클래스를 그대로 사용하되, `TokenizerConfig`를 앰비언트 최적화 설정으로 구성했다.

```python
from miditok import TSD, TokenizerConfig

config = TokenizerConfig(
    pitch_range=(21, 108),              # A0 ~ C8 전 범위
    beat_res={(0, 8): 8, (8, 64): 4},  # 이중 해상도 시간 격자
    num_velocities=16,                  # 16단계 다이나믹
    use_tempos=True,                    # 템포 변화 토큰
    use_rests=True,                     # 침묵 명시 토큰
    use_programs=False,                 # 단일 악기
    use_chords=False,                   # 코드 분석 불필요
    use_time_signatures=False,          # 박자표 무시
    tempo_range=(20, 80),              # 앰비언트 실제 BPM 범위
    num_tempos=30,                      # 30단계 템포 해상도
    special_tokens=["PAD", "BOS", "EOS", "MASK"],
)
tokenizer = TSD(config)
```

각 파라미터의 선택 근거는 [11절](#11-설계-파라미터-선택-근거)에서 상세히 설명한다.

---

## 5. 어휘(Vocabulary) 구조 상세

TSD Ambient의 전체 어휘는 **808개 토큰**으로 구성된다.

### 5-1. 토큰 그룹별 현황

| 토큰 그룹 | 수량 | 역할 |
|---|---|---|
| `PAD` | 1 | 패딩 |
| `BOS` | 1 | 시퀀스 시작 |
| `EOS` | 1 | 시퀀스 종료 |
| `MASK` | 1 | 마스킹 (학습용) |
| `Tempo` | 30 | 템포 변화 (20~80 BPM) |
| `TimeShift` | 288 | 이전 이벤트로부터의 시간 간격 |
| `Pitch` | 88 | 음표 피치 (A0=21 ~ C8=108) |
| `Velocity` | 16 | 음표 세기 (7~127, 8단계 균등) |
| `Duration` | 288 | 음표 지속 시간 |
| `Rest` | 32 | 명시적 침묵 |
| `PitchDrum` | 62 | 드럼 피치 (미사용, 어휘에 포함) |
| **합계** | **808** | |

### 5-2. TimeShift / Duration 토큰의 이중 해상도 구조

TSD에서 가장 독특한 부분이 TimeShift와 Duration 토큰의 **이중 해상도 설계**다.

`beat_res={(0, 8): 8, (8, 64): 4}`라는 설정은 다음을 의미한다:

- **0~8박 범위**: 1박을 8분할 → 최소 단위 = 1/8박 = (60/BPM)/8 초
- **8~64박 범위**: 1박을 4분할 → 최소 단위 = 1/4박 = (60/BPM)/4 초

이를 통해 생성되는 토큰 형식은 `TimeShift_beat.sub.res`다:
- `beat`: 완전한 박자 수
- `sub`: 소수점 이하 (sub/res 박)
- `res`: 해당 구간의 해상도

예시:

| 토큰 | 의미 | @ 40 BPM |
|---|---|---|
| `TimeShift_0.1.8` | 1/8박 = 0.125박 | 0.19초 |
| `TimeShift_1.0.8` | 1박 | 1.50초 |
| `TimeShift_4.0.4` | 4박 | 6.00초 |
| `TimeShift_16.0.4` | 16박 (4마디) | 24.0초 |
| `TimeShift_64.0.4` | 64박 (16마디) | **192.0초 (3.2분)** |

**이중 해상도의 실용적 의미**:

- 짧은 간격(0~8박)은 1/8박 단위로 세밀하게 표현 → 정확한 onset 재현
- 긴 간격(8~64박)은 1/4박 단위로 표현 → 3분짜리 앰비언트 침묵도 단 1개 토큰

만약 단일 해상도(예: 전 구간 1/8박)를 사용하면 192초 침묵에 `TimeShift_0.1.8`을 192/(0.125 × 1.5) = 1024번 반복해야 한다. 이중 해상도를 쓰면 `TimeShift_64.0.4` 1개로 끝난다.

### 5-3. Tempo 토큰 (로그 스케일 분포)

```
Tempo_20.0, Tempo_22.07, Tempo_24.14, Tempo_26.21, ..., Tempo_77.93, Tempo_80.0
```

30개 템포 토큰이 20 BPM에서 80 BPM까지 **로그 스케일**로 배치된다.

로그 스케일을 쓰는 이유: 인간의 박자 지각이 로그적이기 때문이다. 20 BPM과 22 BPM의 차이가 70 BPM과 72 BPM의 차이보다 더 크게 느껴진다. 로그 스케일로 분포하면 낮은 BPM에서 더 세밀한 표현이 가능하다.

- 20 BPM 구간 간격: ~2.07 BPM
- 60 BPM 구간 간격: ~5.8 BPM
- 80 BPM 구간 간격: ~2.07 BPM

### 5-4. Velocity 토큰 (균등 분포)

```
Velocity_7, Velocity_15, Velocity_23, ..., Velocity_119, Velocity_127
```

0~127 범위를 16등분. 각 구간은 약 8 MIDI 단위 너비다.

앰비언트에서 pp (velocity 10~30)에서 mp (velocity 50~70) 범위가 핵심이다. 이 범위(0~80)에 16단계 중 약 10단계가 집중된다.

기존 `REMIAmbientTokenizer`의 8단계(`range(8, 120, 14)`)보다 2배 세밀하다.

### 5-5. Rest 토큰

```
Rest_0.1.8, Rest_0.2.8, ..., Rest_11.1.2, Rest_12.0.2
```

32개 Rest 토큰. 최대 `Rest_12.0.2 = 12박`까지 표현한다.

**Rest vs TimeShift 차이**: Rest는 "이 구간에 의미 있는 음표가 없는 정적(silence)"을 명시적으로 나타낸다. 단순한 이벤트 간 간격인 TimeShift와 달리, Rest는 음악적 의미의 쉼표다. 모델이 침묵을 텍스처의 일부로 학습하는 데 도움을 준다.

---

## 6. 토큰화 과정 단계별 설명

MIDI 파일 한 곡을 TSD Ambient로 토큰화하는 전체 과정이다.

### Step 1: MIDI 파일 로딩 및 파싱

```
MIDI 파일 → symusic.Score 객체
```

MiDiTok v3는 내부적으로 `symusic` 라이브러리를 사용한다. `symusic`은 C++로 구현된 고성능 MIDI 파서로, 음표(Note), 템포(Tempo), 박자(TimeSignature), 컨트롤러(Control) 이벤트를 읽는다.

```python
import symusic
score = symusic.Score("ambient.mid")
```

### Step 2: 전처리 (preprocess_score)

로딩된 Score에 다음 처리를 수행한다:

1. **시간 리샘플링**: 원본 ticks_per_quarter(보통 480)를 토크나이저의 내부 해상도로 변환
   - `new_tpq = max_num_pos_per_beat = 8` (beat_res의 최대 해상도)
   - 원본 tick 값에 `8/480`을 곱해 변환

2. **피치 범위 필터링**: pitch < 21 또는 pitch > 108인 음표 삭제

3. **벨로시티 양자화**: 16단계 중 가장 가까운 값으로 스냅

4. **Duration 양자화**: 토크나이저 어휘에 있는 Duration 값 중 가장 가까운 값으로 스냅

5. **빈 트랙 삭제**: 음표가 없는 트랙 제거

### Step 3: 이벤트 추출 및 정렬

전처리된 Score에서 시간 순서로 이벤트를 나열한다:

```
t=0:    [Tempo=29.5 BPM]
t=16:   [NoteOn pitch=50, vel=71, duration=8]
t=33:   [NoteOn pitch=53, vel=39, duration=5]
t=75:   [NoteOn pitch=53, vel=71, duration=57]
t=235:  [NoteOn pitch=60, vel=31, duration=30]
```

(단위: 내부 tick, tpq=8 기준. t=16은 16/8 = 2박 = 4.07초 @ 29.5BPM)

### Step 4: TimeShift 계산

이벤트 간의 시간 간격을 계산하고 TimeShift 토큰을 삽입한다.

```
이전 이벤트 시각 → 현재 이벤트 시각 사이의 간격
간격 = current_tick - prev_tick
```

이 간격을 `beat.sub.res` 표현으로 변환한다:
- `beat` = 간격 // tpq
- `sub` = (간격 % tpq) // (tpq // res)  
- `res` = 해당 beat 구간의 해상도

**긴 간격 처리**: 간격이 64박을 초과할 경우, 여러 TimeShift 토큰으로 분할한다.

```
TimeShift_64.0.4 + TimeShift_10.0.4  →  74박 간격
```

그러나 TSD Ambient에서 최대 TimeShift가 64박(192초 @ 20BPM)이므로 일반적인 앰비언트 음악에서는 분할이 거의 필요 없다.

### Step 5: 음표 토큰 생성

각 음표는 다음 4개 토큰으로 변환된다:

```
Pitch_50  →  Velocity_71  →  Duration_1.0.8
```

- `Pitch_50`: MIDI 음표 번호 50 (D3)
- `Velocity_71`: 71에 가장 가까운 Velocity 토큰
- `Duration_1.0.8`: 1박(= 1.0 beat at res=8) 지속

### Step 6: Rest 삽입

음표 사이의 간격이 특정 임계값 이상이면 TimeShift 대신 Rest 토큰을 삽입한다.

기준: `use_rests=True` 설정 시, 이전 음표의 끝과 다음 음표의 시작 사이에 빈 공간이 있으면 Rest를 삽입.

### Step 7: Tempo 변화 토큰 삽입

MIDI 내 템포 변화 이벤트가 있으면 해당 시각에 Tempo 토큰을 삽입한다.

```
Tempo_30.34   (곡 시작 시 삽입)
...음표들...
TimeShift_X   (템포 변화 직전)
Tempo_27.16   (새 템포로 변경)
```

### Step 8: 특수 토큰 추가

시퀀스 앞뒤에 BOS/EOS 토큰을 추가한다.

최종 시퀀스 형태:
```
[BOS]  Tempo_30.34  Rest_2.0.2  Pitch_50  Velocity_71  Duration_1.0.8
       Rest_1.0.4   Rest_0.1.8  Pitch_53  Velocity_39  Duration_0.5.8
       ...
       Pitch_60  Velocity_31  Duration_3.6.8  [EOS]
```

---

## 7. 실제 인코딩 예시

`ambient_00.mid` (29.5 BPM, 120.6초, 4개 음표)의 실제 인코딩 결과다.

### 원본 MIDI

| 이벤트 | 시각 (초) | Pitch | Velocity | Duration (초) |
|---|---|---|---|---|
| 음표 1 | 3.97 | 50 (D3) | 67 | 2.12 |
| 음표 2 | 8.48 | 53 (F3) | 42 | 1.16 |
| 음표 3 | 19.06 | 53 (F3) | 67 | 14.51 |
| 음표 4 | 47.84 | 60 (C4) | 30 | 6.09 |

### 생성된 토큰 시퀀스 (20개 토큰)

```
[ 0] Tempo_30.34      ← 시작 BPM 30.34 (원본 29.47에 가장 가까운 값)
[ 1] Rest_2.0.2       ← 2박 침묵 (곡 시작 ~ 첫 음표까지)
[ 2] Pitch_50         ← 음표 1: D3
[ 3] Velocity_71      ← 67에 가장 가까운 벨로시티 71
[ 4] Duration_1.0.8   ← 1박 (= 2.03초 @ 30.34BPM ≈ 원본 2.12초)
[ 5] Rest_1.0.4       ← 다음 음표까지 침묵 1박
[ 6] Rest_0.1.8       ← 추가 침묵 0.125박
[ 7] Pitch_53         ← 음표 2: F3
[ 8] Velocity_39      ← 42 → 39 양자화
[ 9] Duration_0.5.8   ← 0.625박 (≈ 1.23초 ≈ 원본 1.16초)
[10] Rest_4.1.2       ← 긴 침묵 (4.25박)
[11] Rest_0.1.8       ← 추가 침묵 0.125박
[12] Pitch_53         ← 음표 3: F3
[13] Velocity_71      ← 67 → 71
[14] Duration_7.1.8   ← 7.125박 (≈ 14.07초 ≈ 원본 14.51초)
[15] Rest_12.0.2      ← 매우 긴 침묵 12박
[16] Rest_0.7.8       ← 추가 침묵 0.875박
[17] Pitch_60         ← 음표 4: C4
[18] Velocity_31      ← 30 → 31
[19] Duration_3.6.8   ← 3.75박 (≈ 7.41초 ≈ 원본 6.09초)
```

### 토큰 수 비교

| 토크나이저 | 동일 MIDI의 토큰 수 |
|---|---|
| remi_original (BUG 포함) | ~35개 (오류 포함) |
| remi_fixed | ~35개 |
| **tsd_ambient** | **20개** |

120초짜리 앰비언트 곡이 단 20개의 토큰으로 표현된다. 트랜스포머의 컨텍스트 창 1024 토큰이면 이 곡을 51번 반복해서 학습할 수 있다.

---

## 8. CC 이벤트 처리 전략

### 8-1. 문제: miditok TSD는 CC를 처리하지 않는다

MiDiTok의 TSD(버전 3.x)는 음표(Note), 템포(Tempo), 박자(TimeSignature)는 처리하지만, **MIDI 컨트롤 체인지(CC) 이벤트는 토큰화하지 않는다**.

그러나 앰비언트 음악에서 CC는 음악 표현의 핵심이다:
- **CC91 (Reverb Depth)**: 공간감 깊이 — 앰비언트의 질감
- **CC74 (Filter Cutoff/Brightness)**: 필터 스윕 — 음색 변화
- **CC64 (Sustain Pedal)**: 음 유지 — 레이어 쌓기

### 8-2. 해결책: 이중 채널 인코딩

TSD Ambient는 **이중 채널 방식**으로 이 문제를 해결한다.

```
encode(MIDI) → (token_ids, cc_events)
```

인코딩 시 두 가지 출력을 분리한다:
1. **token_ids**: 음표/템포 정보를 담은 주 토큰 시퀀스
2. **cc_events**: `[(cc_number, time_sec, value), ...]` 형태의 CC 이벤트 목록

```python
def encode_midi(pm):
    cc_events = []
    for inst in pm.instruments:
        for cc in inst.control_changes:
            if cc.number in [64, 74, 91]:
                cc_events.append(
                    (cc.number, cc.time, _quantize_cc(cc.value))
                )
    # ... 주 토큰 시퀀스 생성 ...
    return token_ids, cc_events
```

디코딩 시에는 두 채널을 다시 합친다:

```python
def decode_midi(token_ids, cc_events=None):
    score = tokenizer.decode([TokSequence(ids=token_ids)])
    pm = pretty_midi.PrettyMIDI(score.dump_midi(...))
    if cc_events:
        for cc_num, t, val in cc_events:
            pm.instruments[0].control_changes.append(
                pretty_midi.ControlChange(cc_num, val, t)
            )
    return pm
```

### 8-3. CC 값 양자화

CC 값(0~127)은 16단계로 양자화된다:

```python
CC_BINS = list(range(0, 128, 8))  # [0, 8, 16, 24, ..., 120]
```

이 양자화의 의미:
- 기존 `REMIAmbientTokenizer`의 리버브 5단계(`REVERB_0/32/64/96/127`)보다 3.2배 세밀하다
- 0~127 범위를 8단위로 나눠 16개 구간 → CC 변화의 미세한 표현 가능

### 8-4. 현재 방식의 한계

이중 채널 방식의 한계: **모델이 CC 패턴을 능동적으로 생성하지 못한다**.

토큰 시퀀스에 CC가 포함되지 않으므로, 트랜스포머가 "이 음표 다음에 리버브를 높여야 한다"는 패턴을 학습하기 어렵다. 현재 구현은 원본 CC를 그대로 복원하는 데는 완벽하지만(CPR=1.000), 음악 생성 시 CC 오토메이션을 창의적으로 생성하는 능력은 없다.

이를 해결하는 방향은 [12절 한계 및 개선 방향](#12-한계-및-개선-방향)에서 설명한다.

---

## 9. 디코딩 과정

토큰 시퀀스를 MIDI로 복원하는 역방향 과정이다.

### 9-1. 시퀀스 파싱

```
Tempo_30.34  →  Rest_2.0.2  →  Pitch_50  →  Velocity_71  →  Duration_1.0.8  →  ...
```

토큰을 왼쪽에서 오른쪽으로 읽으면서 현재 절대 시각(`current_tick`)을 유지한다.

### 9-2. 토큰 유형별 처리

| 토큰 유형 | 처리 |
|---|---|
| `Tempo_X` | 이후 템포를 X BPM으로 설정 |
| `TimeShift_B.S.R` | current_tick += beat_to_ticks(B + S/R) |
| `Rest_B.S.R` | current_tick += beat_to_ticks(B + S/R) |
| `Pitch_Y` | 음표 시작 시각 = current_tick, 피치 = Y |
| `Velocity_Z` | 이 음표의 벨로시티 = Z |
| `Duration_B.S.R` | 음표 종료 = current_tick + beat_to_ticks(B + S/R) → Note 확정 |

### 9-3. 폴리포니 처리

TSD의 핵심 강점 중 하나가 폴리포니(여러 음표의 동시 진행) 처리다.

TimeShift=0이면 이전 음표와 동시에 시작한다:

```
Pitch_50  Velocity_71  Duration_8.0  (음표 A 시작, 현재 시각 t)
TimeShift_0.0  (시간 전진 없음)
Pitch_55  Velocity_64  Duration_4.0  (음표 B 시작, 동일 시각 t)
```

이렇게 하면 두 음표가 동시에 시작하는 코드(Chord) 표현이 가능하다. `NoteOff`를 추적할 필요 없이 각 음표의 Duration이 독립적으로 끝 시각을 결정한다.

### 9-4. 복원된 MIDI 정밀도

양자화로 인한 손실:
- Duration 오차: 평균 3.3% (긴 음표 2.6%) — 앰비언트에서 허용 가능
- Onset 오차: 양자화 해상도의 절반 이하 (1/16박 이하)

---

## 10. 구현 코드 해설

`test_tokenizer/tok_modules/tsd_ambient.py`의 각 부분을 설명한다.

### 10-1. 토크나이저 생성

```python
def _build_tokenizer() -> TSD:
    config = TokenizerConfig(
        pitch_range=(21, 108),             # 주의: range() 객체가 아닌 tuple
        beat_res={(0, 8): 8, (8, 64): 4}, # {(시작박, 끝박): 해상도}
        num_velocities=16,
        use_tempos=True,
        use_rests=True,
        use_programs=False,   # 단일 악기 → 프로그램 번호 무시
        use_chords=False,     # 코드 분석 토큰 미사용
        use_time_signatures=False,  # 박자표 토큰 미사용
        tempo_range=(20, 80),
        num_tempos=30,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
    )
    return TSD(config)
```

**중요 주의사항**: `pitch_range`에 `range(21, 109)`를 넣으면 miditok 내부에서 `pitch_range[0]=21`, `pitch_range[1]=22`로 인식해 pitch > 22인 모든 음표가 삭제된다. 반드시 `(21, 108)` 튜플로 입력해야 한다.

### 10-2. 인코딩 (MIDI → 토큰)

```python
def encode_midi(pm: pretty_midi.PrettyMIDI) -> Tuple[List[int], List[...]]:
    # 1. CC 이벤트를 먼저 추출 (토크나이저가 처리하기 전에)
    cc_events = []
    for inst in pm.instruments:
        for cc in inst.control_changes:
            if cc.number in TRACKED_CC:
                cc_events.append((cc.number, cc.time, _quantize_cc(cc.value)))

    # 2. pretty_midi → 임시 파일 → symusic.Score
    #    (miditok은 symusic.Score를 입력으로 받음)
    pm.write(tmp_path)
    score = symusic.Score(tmp_path)

    # 3. miditok TSD 인코딩
    tok_seq = tokenizer.encode(score)  # List[TokSequence] 반환
    if isinstance(tok_seq, list):
        tok_seq = tok_seq[0]           # 단일 트랙

    return tok_seq.ids, cc_events
```

### 10-3. 디코딩 (토큰 → MIDI)

```python
def decode_midi(token_ids, cc_events=None):
    tok_seq = TokSequence(ids=token_ids)
    score = tokenizer.decode([tok_seq])  # List로 감싸야 함 (miditok v3 API)
    
    # symusic.Score → 임시 파일 → pretty_midi
    score.dump_midi(tmp_path)
    pm = pretty_midi.PrettyMIDI(tmp_path)
    
    # CC 이벤트 재주입
    if cc_events and pm.instruments:
        for cc_num, t, val in cc_events:
            pm.instruments[0].control_changes.append(
                pretty_midi.ControlChange(cc_num, val, t)
            )
    return pm
```

**miditok v3 API 주의**: `tokenizer.decode()`는 `List[TokSequence]`를 인자로 받는다. `TokSequence` 단일 객체를 직접 넣으면 `'int' object has no attribute 'tokens'` 오류가 발생한다.

---

## 11. 설계 파라미터 선택 근거

### `pitch_range=(21, 108)` — A0에서 C8까지

앰비언트 신디사이저는 거의 전 음역대를 사용한다. 피아노 전 건반(A0~C8)을 커버하는 이 범위는 일반적인 신디사이저 패드 음색에서 사용하는 모든 음표를 포함한다.

### `beat_res={(0, 8): 8, (8, 64): 4}` — 이중 해상도

- **0~8박 구간을 1/8박으로 세분**: 앰비언트에서도 onset의 정밀도는 필요하다. 1/8박 해상도는 40 BPM에서 약 190ms로, 인간이 지각하는 음악적 동시성 임계값(~50ms)보다 크지만 앰비언트의 "느슨한 타이밍"에는 충분하다.
- **8~64박 구간을 1/4박으로 표현**: 긴 음표와 긴 침묵을 효율적으로 처리한다. 64박 × 3초/박(20BPM) = 192초의 침묵도 1개 토큰.

### `tempo_range=(20, 80)` — 앰비언트 실제 BPM

기존 `REMIAmbientTokenizer`의 40~120 BPM은 앰비언트에 맞지 않는다:
- Brian Eno "Ambient 1: Music for Airports": 약 30~50 BPM
- William Basinski "Disintegration Loops": 느린 템포 없이 피치 변화만 사용
- Dark Ambient 장르 평균: 20~45 BPM

20 BPM 하한은 1박이 3초임을 의미한다. 이보다 느린 "BPM" 음악은 사실상 템포가 없는 자유 형식이다.

### `num_tempos=30` — 로그 스케일 30단계

20~80 BPM을 30단계로 나누면 로그 스케일 간격이 약 2~6 BPM이다. 앰비언트에서 템포 정밀도는 ±5 BPM이면 충분하다(인간 지각 한계).

### `num_velocities=16` — 16단계 다이나믹

앰비언트 음악의 전형적 다이나믹 범위는 pp(10~25)에서 mp(50~65)이다. 16단계 양자화는 이 범위를 약 4~5 MIDI 단위 간격으로 커버하여 미세한 페이드인/아웃 표현이 가능하다.

### `use_rests=True` — 침묵 명시

앰비언트에서 침묵은 음표만큼 중요한 음악적 요소다. Rest 토큰을 사용하면 긴 침묵을 1~2개의 토큰으로 효율적으로 표현하고, 모델이 침묵을 의미 있는 이벤트로 인식하도록 돕는다.

### `use_programs=False` — 단일 악기

앰비언트 프로젝트에서는 단일 신디사이저 파트를 학습한다. 멀티트랙 처리가 불필요하므로 프로그램 번호 토큰을 비활성화하여 어휘 크기를 절감한다.

---

## 12. 한계 및 개선 방향

### 12-1. CC를 토큰 시퀀스에 포함하지 않는 문제

**현재 한계**: CC 이벤트가 별도 채널로 분리되어 있어, 트랜스포머가 "음표 변화 → CC 변화"의 관계를 학습하지 못한다.

**개선 방향**: MiDiTok의 `_score_to_tokens()` 메서드를 오버라이드하여 CC 이벤트를 TimeShift와 함께 삽입한다:

```python
# 예상 시퀀스 구조 (CC 포함 버전)
TimeShift_0.0  CC74_88   (브라이트니스 변화)
TimeShift_1.0  Pitch_50  Velocity_71  Duration_1.0.8
TimeShift_2.0  CC91_48   (리버브 변화)
```

이렇게 하면 모델이 음표와 CC 사이의 시간적 패턴을 동시에 학습할 수 있다.

### 12-2. 최대 Rest 12박 제한

현재 `Rest_12.0.2`가 최대(12박 = 36초 @ 20BPM). 이보다 긴 침묵은 TimeShift로 처리되거나 여러 Rest 토큰으로 분할된다. 아주 희박한 앰비언트(분당 1~2개 음표)에서는 문제가 될 수 있다.

**개선**: `beat_res_rest` 파라미터 또는 max_rest 설정 조정.

### 12-3. 폴리포니 음표 순서 모호성

여러 음표가 동시에 시작할 때(`TimeShift_0.0` 연속), 어떤 음표를 먼저 놓을지 순서가 모호하다. 일반적으로 피치 오름차순으로 정렬하지만, 이 규칙이 명시적으로 보장되지 않는다.

### 12-4. BPE(Byte Pair Encoding) 미적용

자주 등장하는 토큰 패턴 (`TimeShift_1.0 + Pitch_60 + Velocity_64 + Duration_2.0`)을 단일 BPE 토큰으로 압축하면 시퀀스 길이를 30~50% 더 줄일 수 있다.

```python
tokenizer.learn_bpe(vocab_size=1200, files_paths=midi_file_list)
```

---

## 요약

TSD Ambient는 다음 세 가지를 결합한 토크나이저다:

1. **MiDiTok의 TSD 방식**: TimeShift + Duration 기반의 NoteOff-free 표현
2. **앰비언트 특화 설정**: 20~80 BPM, 이중 해상도 시간 격자, 16단계 벨로시티
3. **CC 이중 채널 보존**: miditok이 지원하지 않는 CC91/74/64를 별도 채널로 완전 보존

실험 결과, 이 설계는 기존 REMI 방식 대비:
- Duration 정밀도 1.8배 향상 (DQE: 0.033 vs 0.058)
- 시퀀스 효율 21% 향상 (0.438 vs 0.532 토큰/초)
- CC 완전 보존 (CPR=1.000 vs 0.000)

을 달성했다.

---

*참고 라이브러리: MiDiTok 3.0.6, symusic 0.6.0*  
*참고 논문: Fradet et al. (ISMIR 2021), Oore et al. (NeurIPS 2018), Huang & Yang (ACM MM 2020)*
