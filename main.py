"""
AmbientFlow 메인 CLI

스타일 폴더 모드 (권장):
  python main.py --style-dir styles/ --input melody.mid
  python main.py --style-dir styles/ --style ballad.mid cinematic.mid

기존 모드 (입력에서 테마 추출):
  python main.py --input melody.mid
  python main.py --realtime --port "MIDI Controller"
"""
import click
import threading
from typing import Optional, List

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    USE_RICH = True
except ImportError:
    console = None
    USE_RICH = False


def _print(msg: str):
    if USE_RICH:
        console.print(msg)
    else:
        print(msg)


@click.command()
# ── 스타일 옵션 ────────────────────────────────────────────────────────
@click.option("--style-dir", "style_dir", default=None,
              help="스타일 MIDI 폴더 경로 (지정하면 해당 폴더에서 테마 로드)")
@click.option("--style", "style_names", multiple=True, default=None,
              help="사용할 스타일 파일명 (여러 개 가능). 미지정 시 폴더 전체 블렌딩")
@click.option("--blend", "blend_strategy",
              type=click.Choice(["front", "center", "random"]),
              default="front",
              help="블렌딩 전략: front(앞부분), center(중간), random(랜덤)")
# ── 입력 옵션 ──────────────────────────────────────────────────────────
@click.option("--input", "-i", "input_path", default=None,
              help="입력 멜로디 파일 (MIDI/오디오). style-dir 모드: context 시드로 사용")
@click.option("--realtime", is_flag=True, default=False,
              help="실시간 MIDI 입력 모드")
@click.option("--port", default=None,
              help="MIDI 포트 이름 (--realtime과 함께 사용)")
# ── 모델/출력 옵션 ────────────────────────────────────────────────────
@click.option("--output", "-o", "output_path", default=None,
              help="출력 파일 경로 (None이면 실시간 재생)")
@click.option("--model", "-m", default="checkpoints/best_model.pt",
              help="생성 모델 체크포인트")
@click.option("--encoder", "-e", default=None,
              help="인코더 체크포인트 (기존 모드 전용)")
@click.option("--device", default="auto",
              type=click.Choice(["auto", "cuda", "cpu"]),
              help="연산 장치")
def main(style_dir, style_names, blend_strategy,
         input_path, realtime, port,
         output_path, model, encoder, device):
    """AmbientFlow: 스타일 기반 실시간 앰비언트 음악 생성기"""

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _print(f"\n[AmbientFlow] 장치: {device}")

    from src.streaming.engine import AmbientFlowEngine
    engine = AmbientFlowEngine(
        model_ckpt=model,
        encoder_ckpt=encoder,
        style_dir=style_dir,
        output_file=output_path,
        device=device,
    )
    engine.start()

    # ── 스타일 설정 ─────────────────────────────────────────────────
    if style_dir:
        names = list(style_names) if style_names else None
        engine.set_style(names=names, strategy=blend_strategy)

        if input_path:
            engine.feed_melody(path=input_path)
        elif realtime:
            _print(f"실시간 MIDI 멜로디 수집: {port or '첫 번째 포트'} (4초)")
            engine.feed_melody(realtime_port=port, realtime_sec=4.0)
        else:
            _print("스타일만 설정됨 — 'm' 명령으로 멜로디 입력 가능")

    # ── 기존 모드 ────────────────────────────────────────────────────
    else:
        if input_path:
            engine.feed_input(path=input_path)
        elif realtime:
            engine.feed_input(realtime_port=port, realtime_sec=4.0)
        else:
            _print("입력 없이 시작 — 'f' 명령으로 파일 로드 가능")

    _print("\n명령어:")
    if style_dir:
        _print("  [s]tyle [파일명...]  스타일 변경 (미지정 시 전체 블렌딩)")
        _print("  [m]elody [파일]      새 멜로디 시드 입력")
        _print("  [l]ist               스타일 목록 출력")
        _print("  [r]eload             styles/ 폴더 재스캔")
    else:
        _print("  [f]ile [파일]        새 입력 파일")
        _print("  [r]ealtime           실시간 MIDI 수집")
    _print("  [q]uit               종료\n")

    _interactive_loop(engine, style_dir, port, blend_strategy)


def _interactive_loop(engine, style_dir: Optional[str],
                      default_port: Optional[str],
                      blend_strategy: str):
    while True:
        try:
            cmd = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not cmd:
            continue

        parts = cmd.split()
        verb = parts[0].lower()

        # ── 스타일 폴더 모드 명령 ──────────────────────────────────
        if style_dir:
            if verb in ("q", "quit", "exit"):
                break

            elif verb in ("s", "style"):
                names = parts[1:] if len(parts) > 1 else None
                try:
                    engine.set_style(names=names, strategy=blend_strategy)
                except Exception as e:
                    _print(f"오류: {e}")

            elif verb in ("m", "melody"):
                if len(parts) > 1:
                    engine.feed_melody(path=parts[1])
                else:
                    path = input("멜로디 파일 경로: ").strip()
                    engine.feed_melody(path=path)

            elif verb in ("l", "list"):
                names = engine.list_styles()
                _print(f"사용 가능한 스타일 ({len(names)}개):")
                for n in names:
                    _print(f"  - {n}")

            elif verb in ("r", "reload"):
                engine.reload_styles()
                _print(f"재로드 완료: {engine.list_styles()}")

            elif verb in ("mr", "melody-realtime"):
                _print("실시간 MIDI 멜로디 수집 시작 (4초)...")
                threading.Thread(
                    target=lambda: engine.feed_melody(
                        realtime_port=default_port, realtime_sec=4.0
                    ), daemon=True
                ).start()

            else:
                _print(f"알 수 없는 명령: {cmd}")

        # ── 기존 모드 명령 ─────────────────────────────────────────
        else:
            if verb in ("q", "quit", "exit"):
                break
            elif verb in ("f", "file"):
                path = parts[1] if len(parts) > 1 else input("파일 경로: ").strip()
                engine.feed_input(path=path)
            elif verb in ("r", "realtime"):
                _print("MIDI 수집 시작 (4초)...")
                threading.Thread(
                    target=lambda: engine.feed_input(
                        realtime_port=default_port, realtime_sec=4.0
                    ), daemon=True
                ).start()
            else:
                _print(f"알 수 없는 명령: {cmd}")

    engine.stop()
    _print("종료")


if __name__ == "__main__":
    main()
