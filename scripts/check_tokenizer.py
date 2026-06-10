import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
from pathlib import Path
import pretty_midi
from src.tokenizer import TSDTokenizer

def main():
    parser = argparse.ArgumentParser(description="TSD Tokenizer Check Script")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing MIDI or .pkl files")
    parser.add_argument("--file_path", type=str, default=None, help="Specific MIDI or .pkl file to tokenize/decode")
    args = parser.parse_args()

    tokenizer = TSDTokenizer()
    target_file = None
    is_pkl = False

    # 1. Determine the target file
    if args.file_path:
        target_file = Path(args.file_path)
        if not target_file.exists():
            print(f"[ERROR] Specified file not found: {target_file}")
            return
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"[ERROR] Specified directory not found: {data_dir}")
            return
        
        # Search for MIDI files first
        midi_files = sorted(list(data_dir.glob("**/*.mid")) + list(data_dir.glob("**/*.midi")))
        if midi_files:
            target_file = midi_files[0]
            print(f"Found {len(midi_files)} MIDI file(s) in {data_dir}. Using the first one: {target_file.name}")
        else:
            # Search for .pkl files
            pkl_files = sorted(list(data_dir.glob("**/*.pkl")))
            if pkl_files:
                target_file = pkl_files[0]
                is_pkl = True
                print(f"Found {len(pkl_files)} .pkl file(s) in {data_dir}. Using the first one: {target_file.name}")
            else:
                print(f"[ERROR] No MIDI (.mid, .midi) or preprocessed (.pkl) files found in {data_dir}")
                return
    else:
        # Default fallback to examples/theme.mid
        default_path = Path("examples/theme.mid")
        if default_path.exists():
            target_file = default_path
            print(f"No arguments provided. Falling back to default: {default_path}")
        else:
            print("[ERROR] Please provide --data_dir or --file_path.")
            return

    # 2. Tokenize (MIDI) or Decode (pkl)
    if target_file.suffix.lower() == '.pkl' or is_pkl:
        print(f"Loading preprocessed tokens from: {target_file}")
        with open(target_file, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "tokens" in data:
            tokens = data["tokens"]
        elif isinstance(data, list):
            tokens = data
        else:
            print("[ERROR] Invalid .pkl format. Expected list of tokens or dict with 'tokens' key.")
            return
    else:
        print(f"Loading MIDI file: {target_file}")
        try:
            midi = pretty_midi.PrettyMIDI(str(target_file))
        except Exception as e:
            print(f"[ERROR] Failed to load MIDI file: {e}")
            return
        
        print("Tokenizing MIDI...")
        tokens = tokenizer.midi_to_tokens(midi)

    # 3. Print Token info
    print(f"Total tokens: {len(tokens)}")
    
    # Print first 50 tokens
    print("\n--- Tokens (ID and String) ---")
    token_strs = tokenizer.decode(tokens)
    for idx, (tid, tstr) in enumerate(zip(tokens, token_strs)):
        print(f"{idx:02d}: ID={tid:<5} -> {tstr}")
        
    # Count token type frequencies
    print("\n--- Token Types Counts ---")
    types = {}
    for tstr in token_strs:
        prefix = tstr.split("_")[0] if "_" in tstr else tstr
        types[prefix] = types.get(prefix, 0) + 1
    for k, v in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:<15} : {v} occurrences")

if __name__ == "__main__":
    main()
