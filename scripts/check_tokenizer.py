import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pretty_midi
from src.tokenizer import TSDTokenizer

def main():
    tokenizer = TSDTokenizer()
    midi_path = "examples/theme.mid"
    if not os.path.exists(midi_path):
        print(f"[ERROR] MIDI file not found: {midi_path}")
        return
        
    print(f"Loading MIDI file: {midi_path}")
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    print("Tokenizing...")
    tokens = tokenizer.midi_to_tokens(midi)
    print(f"Total tokens: {len(tokens)}")
    
    # Print first 50 tokens
    print("\n--- First 50 Tokens (ID and String) ---")
    token_strs = tokenizer.decode(tokens)
    for idx, (tid, tstr) in enumerate(zip(tokens[:50], token_strs[:50])):
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
