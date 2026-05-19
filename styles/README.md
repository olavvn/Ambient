# styles/ — 스타일 MIDI 폴더

이 폴더에 원하는 스타일의 MIDI 파일을 넣어두면, AmbientFlow가 해당 파일들을
블렌딩하여 생성 모델의 **테마 컨디션**으로 사용합니다.

입력 멜로디(`--input`)는 테마와 독립적으로 **생성 컨텍스트 시드**로만 사용됩니다.

## 빠른 시작

```bash
# styles/ 폴더 전체 블렌딩
python main.py --style-dir styles/ --input melody.mid

# 특정 파일만 선택
python main.py --style-dir styles/ --style ballad.mid cinematic.mid

# 블렌딩 전략 선택 (front / center / random)
python main.py --style-dir styles/ --blend center
```

## 파일 추가 방법

1. `.mid` 또는 `.midi` 파일을 이 폴더에 복사합니다.
2. AmbientFlow를 시작하거나, 실행 중이라면 `r` (reload) 명령을 입력합니다.

```
> r          ← styles/ 폴더 재스캔 (핫 리로드)
> l          ← 현재 로드된 스타일 목록 확인
> s ballad.mid dark.mid   ← 특정 스타일로 전환
```

## 권장 파일 구성

| 파일명 | 설명 |
|--------|------|
| `ballad.mid` | 서정적 발라드 느낌 |
| `cinematic.mid` | 영화 음악 느낌 |
| `dark.mid` | 어두운 분위기 |
| `ambient.mid` | 순수 앰비언트 |
| `jazz.mid` | 재즈 스타일 |

파일명은 자유롭게 지정할 수 있습니다.

## 블렌딩 방식

여러 파일을 사용할 때, 각 파일에서 동일한 길이(`max_theme_len / 파일 수`)의
구간을 잘라내어 순서대로 이어붙입니다.

| 전략 | 설명 |
|------|------|
| `front` (기본) | 각 파일의 앞부분 사용 |
| `center` | 각 파일의 중간 부분 사용 |
| `random` | 각 파일의 무작위 구간 사용 |

## 주의사항

- MIDI 파일은 최소 1개 이상 있어야 합니다. 없으면 `FileNotFoundError`가 발생합니다.
- 파일이 너무 짧으면 해당 파일에서 뽑을 수 있는 구간이 줄어들 수 있습니다.
- 권장 파일 길이: 32마디 이상 (약 60초 이상의 MIDI)
