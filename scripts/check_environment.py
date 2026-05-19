"""환경 검증 스크립트"""
import sys


def check():
    checks = []

    import torch
    cuda_ok = torch.cuda.is_available()
    checks.append(("PyTorch CUDA", cuda_ok, torch.version.cuda or "N/A"))

    try:
        import pretty_midi, mido
        checks.append(("pretty_midi", True, pretty_midi.__version__))
    except ImportError as e:
        checks.append(("pretty_midi", False, str(e)))

    try:
        import librosa
        checks.append(("librosa", True, librosa.__version__))
    except ImportError as e:
        checks.append(("librosa", False, str(e)))

    try:
        import basic_pitch
        checks.append(("basic-pitch", True, "ok"))
    except ImportError as e:
        checks.append(("basic-pitch", False, str(e)))

    try:
        import fluidsynth
        checks.append(("pyfluidsynth", True, "ok"))
    except Exception as e:
        checks.append(("pyfluidsynth", False, str(e)))

    try:
        import sklearn
        checks.append(("scikit-learn", True, sklearn.__version__))
    except ImportError as e:
        checks.append(("scikit-learn", False, str(e)))

    try:
        import einops
        checks.append(("einops", True, einops.__version__))
    except ImportError as e:
        checks.append(("einops", False, str(e)))

    print("\n=== Environment Check ===")
    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")

    failed = [c for c in checks if not c[1]]
    if failed:
        print(f"\n{len(failed)} check(s) failed. Please review setup.")
        sys.exit(1)
    else:
        print("\nAll checks passed! Ready to run AmbientFlow.")


if __name__ == "__main__":
    check()
