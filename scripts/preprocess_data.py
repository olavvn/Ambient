"""MIDI 전처리 스크립트"""
import pickle
import pathlib
import argparse
import pretty_midi
from src.theme.tokenizer import REMIAmbientTokenizer
from src.theme.extractor import ThemeExtractor


def process_file(midi_path: str, tokenizer: REMIAmbientTokenizer,
                 extractor: ThemeExtractor) -> dict:
    midi = pretty_midi.PrettyMIDI(midi_path)
    theme_spans = extractor.extract_theme_spans(midi)
    tokens = tokenizer.midi_to_tokens(midi, theme_spans=theme_spans)
    chunks = [tokens[i:i + 512] for i in range(0, len(tokens) - 512, 256)]
    return {
        "path": midi_path,
        "tokens": tokens,
        "chunks": chunks,
        "theme_spans": theme_spans,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw_midi/")
    parser.add_argument("--output", default="data_pkl/")
    args = parser.parse_args()

    tokenizer = REMIAmbientTokenizer()
    extractor = ThemeExtractor()

    data_dir = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(exist_ok=True)

    midi_files = list(data_dir.glob("**/*.mid")) + list(data_dir.glob("**/*.midi"))
    print(f"처리할 MIDI 파일: {len(midi_files)}개")

    for f in midi_files:
        try:
            result = process_file(str(f), tokenizer, extractor)
            out_path = out_dir / (f.stem + ".pkl")
            with open(out_path, "wb") as fp:
                pickle.dump(result, fp)
            print(f"처리 완료: {f.name} → {len(result['chunks'])} 청크")
        except Exception as e:
            print(f"오류: {f.name} — {e}")


if __name__ == "__main__":
    main()
