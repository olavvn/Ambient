"""
Ambient test MIDI generator - creates 20 ambient-style MIDI files for tokenizer evaluation
"""
import mido
import numpy as np
import os
import random

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_midi")
os.makedirs(OUTPUT_DIR, exist_ok=True)

AMBIENT_SCALES = {
    "C_major": [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72],
    "F_major": [41, 43, 45, 46, 48, 50, 52, 53, 55, 57, 58, 60, 62, 64, 65],
    "G_major": [43, 45, 47, 48, 50, 52, 54, 55, 57, 59, 60, 62, 64, 66, 67],
    "A_minor": [45, 47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69],
    "D_minor": [38, 40, 41, 43, 45, 46, 48, 50, 52, 53, 55, 57, 58, 60, 62],
    "E_minor": [40, 42, 43, 45, 47, 48, 50, 52, 54, 55, 57, 59, 60, 62, 64],
}

TICKS_PER_BEAT = 480


def bpm_to_tempo(bpm: float) -> int:
    return int(60_000_000 / bpm)


def seconds_to_ticks(seconds: float, bpm: float) -> int:
    return int(seconds * (bpm / 60.0) * TICKS_PER_BEAT)


def generate_ambient_midi(
    duration_sec: float,
    bpm: float,
    scale_name: str,
    note_density: float = 0.3,
    reverb_variation: bool = True,
    brightness_variation: bool = True,
    seed: int = 0,
) -> mido.MidiFile:
    rng = np.random.RandomState(seed)
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Tempo track (track 0 convention)
    track.append(mido.MetaMessage("set_tempo", tempo=bpm_to_tempo(bpm), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    # Add a few subtle tempo changes
    n_tempo_changes = rng.randint(2, 5)
    tempo_change_times = sorted(rng.uniform(duration_sec * 0.2, duration_sec * 0.8, n_tempo_changes))
    for t in tempo_change_times:
        new_bpm = float(np.clip(bpm + rng.uniform(-5, 5), 20, 80))
        tick = seconds_to_ticks(t, bpm)
        track.append(mido.MetaMessage("set_tempo", tempo=bpm_to_tempo(new_bpm), time=0))

    scale = AMBIENT_SCALES[scale_name]

    # Collect all events as (time_sec, msg)
    events = []

    # Notes: log-uniform duration, sparse density
    t = 0.0
    while t < duration_sec - 2.0:
        gap = rng.exponential(60.0 / bpm * 2)
        if rng.random() > note_density:
            t += gap
            continue
        note_dur = float(np.exp(rng.uniform(np.log(1.0), np.log(min(25.0, duration_sec - t - 1.0)))))
        pitch = int(rng.choice(scale))
        velocity = int(rng.uniform(20, 80))
        end = t + note_dur
        if end > duration_sec - 0.1:
            end = duration_sec - 0.1
        if end <= t + 0.5:
            t += gap
            continue
        events.append((t, mido.Message("note_on", channel=0, note=pitch, velocity=velocity, time=0)))
        events.append((end, mido.Message("note_on", channel=0, note=pitch, velocity=0, time=0)))
        t += gap

    # CC91 (Reverb)
    if reverb_variation:
        reverb_times = np.linspace(0, duration_sec, 20)
        base_reverb = rng.randint(40, 120)
        for rt in reverb_times:
            val = int(np.clip(base_reverb + rng.randint(-20, 20), 0, 127))
            events.append((float(rt), mido.Message("control_change", channel=0, control=91, value=val, time=0)))

    # CC74 (Brightness)
    if brightness_variation:
        bright_times = np.linspace(0, duration_sec, 15)
        for bt in bright_times:
            val = int(rng.uniform(20, 100))
            events.append((float(bt), mido.Message("control_change", channel=0, control=74, value=val, time=0)))

    # CC64 (Sustain)
    for _ in range(3):
        on_t = float(rng.uniform(0, duration_sec * 0.7))
        off_t = on_t + 60.0 / bpm * 2
        events.append((on_t, mido.Message("control_change", channel=0, control=64, value=127, time=0)))
        events.append((off_t, mido.Message("control_change", channel=0, control=64, value=0, time=0)))

    # Sort by time and convert to delta ticks
    events.sort(key=lambda x: x[0])
    prev_tick = 0
    for t_sec, msg in events:
        tick = seconds_to_ticks(t_sec, bpm)
        delta = max(0, tick - prev_tick)
        track.append(msg.copy(time=delta))
        prev_tick = tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid


def main():
    configs = []
    scales = list(AMBIENT_SCALES.keys())
    for i in range(20):
        duration = random.uniform(60, 180)
        bpm = random.uniform(25, 65)
        scale = scales[i % len(scales)]
        density = random.uniform(0.15, 0.5)
        configs.append((duration, bpm, scale, density))

    for i, (duration, bpm, scale, density) in enumerate(configs):
        mid = generate_ambient_midi(
            duration_sec=duration,
            bpm=bpm,
            scale_name=scale,
            note_density=density,
            reverb_variation=True,
            brightness_variation=True,
            seed=i,
        )
        out_path = os.path.join(OUTPUT_DIR, f"ambient_{i:02d}.mid")
        mid.save(out_path)
        print(f"Generated {out_path}: {duration:.0f}s, {bpm:.1f} BPM, {scale}")

    print(f"\nGenerated {len(configs)} MIDI files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
