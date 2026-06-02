"""
Tokenizer evaluation script.
Compares remi_original, remi_fixed, and tsd_ambient on 6 metrics.
"""
from __future__ import annotations
import os
import sys
import glob
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mir_eval
import pretty_midi

warnings.filterwarnings("ignore")

# Ensure test_tokenizer package root is on path
sys.path.insert(0, os.path.dirname(__file__))

from tok_modules import remi_original, remi_fixed, tsd_ambient

MIDI_DIR = os.path.join(os.path.dirname(__file__), "test_midi")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def pm_to_note_arrays(pm: pretty_midi.PrettyMIDI):
    """Return (onsets, offsets, pitches) arrays from all non-drum instruments."""
    onsets, offsets, pitches = [], [], []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            onsets.append(note.start)
            offsets.append(note.end)
            pitches.append(float(note.pitch))
    if not onsets:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(onsets)
    return np.array(onsets)[order], np.array(offsets)[order], np.array(pitches)[order]


def reconstruction_fidelity(orig: pretty_midi.PrettyMIDI,
                             recon: pretty_midi.PrettyMIDI) -> dict:
    ref_on, ref_off, ref_p = pm_to_note_arrays(orig)
    est_on, est_off, est_p = pm_to_note_arrays(recon)

    if len(ref_on) == 0 or len(est_on) == 0:
        return {"RF_pitch": 0.0, "RF_onset": 0.0, "RF_f1": 0.0,
                "RF_precision": 0.0, "RF_recall": 0.0}

    ref_intervals = np.column_stack([ref_on, ref_off])
    est_intervals = np.column_stack([est_on, est_off])

    # onset+offset+pitch matching
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_p,
        est_intervals, est_p,
        onset_tolerance=0.05,
        offset_ratio=0.2,
        offset_min_tolerance=0.05,
        pitch_tolerance=0.5,
    )

    # onset-only matching (pitch must match)
    p_on, r_on, f_on, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_p,
        est_intervals, est_p,
        onset_tolerance=0.05,
        offset_ratio=None,
        pitch_tolerance=0.5,
    )

    return {
        "RF_pitch": float(f_on),    # onset+pitch F1
        "RF_onset": float(f_on),
        "RF_f1": float(f),          # onset+offset+pitch F1
        "RF_precision": float(p),
        "RF_recall": float(r),
    }


def sequence_length_efficiency(token_ids, pm_orig: pretty_midi.PrettyMIDI) -> dict:
    end_time = pm_orig.get_end_time()
    if end_time <= 0:
        return {"SLE_tokens_per_sec": 0.0, "SLE_total_tokens": len(token_ids)}
    return {
        "SLE_tokens_per_sec": len(token_ids) / end_time,
        "SLE_total_tokens": len(token_ids),
    }


def cc_preservation_rate(orig: pretty_midi.PrettyMIDI,
                          recon: pretty_midi.PrettyMIDI,
                          cc_numbers=(64, 74, 91)) -> dict:
    results = {}
    for cc_num in cc_numbers:
        orig_events = []
        recon_events = []
        for inst in orig.instruments:
            orig_events += [(cc.time, cc.value) for cc in inst.control_changes if cc.number == cc_num]
        for inst in recon.instruments:
            recon_events += [(cc.time, cc.value) for cc in inst.control_changes if cc.number == cc_num]

        if not orig_events:
            results[f"CPR_CC{cc_num}"] = float("nan")
            continue

        # Simple time-based matching: count how many orig events have a match within 200ms
        orig_times = np.array([e[0] for e in orig_events])
        recon_times = np.array([e[0] for e in recon_events]) if recon_events else np.array([])

        matched = 0
        for ot in orig_times:
            if len(recon_times) > 0 and np.min(np.abs(recon_times - ot)) <= 0.2:
                matched += 1
        results[f"CPR_CC{cc_num}"] = matched / len(orig_times)

    return results


def duration_quantization_error(orig: pretty_midi.PrettyMIDI,
                                  recon: pretty_midi.PrettyMIDI) -> dict:
    ref_on, ref_off, ref_p = pm_to_note_arrays(orig)
    est_on, est_off, est_p = pm_to_note_arrays(recon)

    if len(ref_on) == 0 or len(est_on) == 0:
        return {"DQE_mean": float("nan"), "DQE_long_notes": float("nan")}

    # Match notes by pitch+onset proximity
    errors = []
    long_errors = []
    ref_durs = ref_off - ref_on

    for i, (ro, rp, rd) in enumerate(zip(ref_on, ref_p, ref_durs)):
        if len(est_on) == 0:
            break
        # Find closest onset with matching pitch
        mask = est_p == rp
        if not np.any(mask):
            continue
        est_on_matched = est_on[mask]
        est_off_matched = est_off[mask]
        closest_idx = np.argmin(np.abs(est_on_matched - ro))
        if np.abs(est_on_matched[closest_idx] - ro) > 0.5:
            continue
        ed = est_off_matched[closest_idx] - est_on_matched[closest_idx]
        rel_err = abs(rd - ed) / max(rd, 0.01)
        errors.append(rel_err)
        if rd >= 2.0:
            long_errors.append(rel_err)

    return {
        "DQE_mean": float(np.mean(errors)) if errors else float("nan"),
        "DQE_long_notes": float(np.mean(long_errors)) if long_errors else float("nan"),
    }


def tempo_range_coverage(orig: pretty_midi.PrettyMIDI,
                          recon: pretty_midi.PrettyMIDI) -> dict:
    orig_times, orig_tempos = orig.get_tempo_changes()
    recon_times, recon_tempos = recon.get_tempo_changes()

    if len(orig_tempos) == 0:
        return {"TRC": float("nan")}

    represented = 0
    for ot in orig_tempos:
        if len(recon_tempos) > 0:
            closest = recon_tempos[np.argmin(np.abs(np.array(recon_tempos) - ot))]
            if abs(closest - ot) / max(ot, 1) < 0.1:
                represented += 1

    return {"TRC": represented / len(orig_tempos)}


def vocabulary_utilization(token_ids_all_files: list, vocab_size: int) -> dict:
    all_tokens = []
    for ids in token_ids_all_files:
        all_tokens.extend(ids)
    unique = len(set(all_tokens))
    return {"VU": unique / max(vocab_size, 1), "VU_unique": unique, "VU_total_vocab": vocab_size}


# ── Evaluation runners ───────────────────────────────────────────────────────

def evaluate_remi_original(midi_files):
    tok = remi_original.REMIAmbientTokenizer()
    per_file = []
    all_token_ids = []

    for path in midi_files:
        try:
            pm = pretty_midi.PrettyMIDI(path)
            initial_tempo = pm.get_tempo_changes()[1][0] if len(pm.get_tempo_changes()[1]) > 0 else 60.0
            token_ids = tok.encode_midi(pm)
            pm_recon = tok.decode_midi(token_ids, bpm=initial_tempo)
            all_token_ids.append(token_ids)

            metrics = {}
            metrics["file"] = os.path.basename(path)
            metrics.update(reconstruction_fidelity(pm, pm_recon))
            metrics.update(sequence_length_efficiency(token_ids, pm))
            metrics.update(cc_preservation_rate(pm, pm_recon))
            metrics.update(duration_quantization_error(pm, pm_recon))
            metrics.update(tempo_range_coverage(pm, pm_recon))
            per_file.append(metrics)
        except Exception as e:
            print(f"  [remi_original] Error on {os.path.basename(path)}: {e}")

    vu = vocabulary_utilization(all_token_ids, tok.vocab_size)
    return per_file, vu


def evaluate_remi_fixed(midi_files):
    per_file = []
    all_token_ids = []

    for path in midi_files:
        try:
            pm = pretty_midi.PrettyMIDI(path)
            token_ids = remi_fixed.encode_midi(pm)
            pm_recon = remi_fixed.decode_midi(token_ids)
            all_token_ids.append(token_ids)

            metrics = {}
            metrics["file"] = os.path.basename(path)
            metrics.update(reconstruction_fidelity(pm, pm_recon))
            metrics.update(sequence_length_efficiency(token_ids, pm))
            metrics.update(cc_preservation_rate(pm, pm_recon))
            metrics.update(duration_quantization_error(pm, pm_recon))
            metrics.update(tempo_range_coverage(pm, pm_recon))
            per_file.append(metrics)
        except Exception as e:
            print(f"  [remi_fixed] Error on {os.path.basename(path)}: {e}")

    vu = vocabulary_utilization(all_token_ids, remi_fixed.tokenizer.vocab_size)
    return per_file, vu


def evaluate_tsd_ambient(midi_files):
    per_file = []
    all_token_ids = []

    for path in midi_files:
        try:
            pm = pretty_midi.PrettyMIDI(path)
            token_ids, cc_events = tsd_ambient.encode_midi(pm)
            pm_recon = tsd_ambient.decode_midi(token_ids, cc_events)
            all_token_ids.append(token_ids)

            metrics = {}
            metrics["file"] = os.path.basename(path)
            metrics.update(reconstruction_fidelity(pm, pm_recon))
            metrics.update(sequence_length_efficiency(token_ids, pm))
            metrics.update(cc_preservation_rate(pm, pm_recon))
            metrics.update(duration_quantization_error(pm, pm_recon))
            metrics.update(tempo_range_coverage(pm, pm_recon))
            per_file.append(metrics)
        except Exception as e:
            print(f"  [tsd_ambient] Error on {os.path.basename(path)}: {e}")

    vu = vocabulary_utilization(all_token_ids, tsd_ambient.tokenizer.vocab_size)
    return per_file, vu


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(summary_df: pd.DataFrame, per_file_df: pd.DataFrame):
    sns.set_theme(style="whitegrid", palette="muted")

    # Figure 1: RF comparison radar-style bar chart
    rf_cols = ["RF_f1", "RF_precision", "RF_recall"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(rf_cols))
    width = 0.25
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    tokenizers = ["remi_original", "remi_fixed", "tsd_ambient"]
    for i, (tok, color) in enumerate(zip(tokenizers, colors)):
        row = summary_df[summary_df["tokenizer"] == tok]
        if row.empty:
            continue
        vals = [row[c].values[0] for c in rf_cols]
        ax.bar(x + i * width, vals, width, label=tok, color=color, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["RF F1", "RF Precision", "RF Recall"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 1: Reconstruction Fidelity Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig1_rf_comparison.png"), dpi=120)
    plt.close(fig)

    # Figure 2: Duration quantization error boxplot
    dqe_data = []
    for tok in tokenizers:
        sub = per_file_df[per_file_df["tokenizer"] == tok]["DQE_mean"].dropna()
        for v in sub:
            dqe_data.append({"tokenizer": tok, "DQE_mean": v})
    if dqe_data:
        dqe_df = pd.DataFrame(dqe_data)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=dqe_df, x="tokenizer", y="DQE_mean", palette=colors, ax=ax)
        ax.set_title("Figure 2: Duration Quantization Error (lower is better)")
        ax.set_ylabel("Mean Absolute Relative Error")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "fig2_dqe_boxplot.png"), dpi=120)
        plt.close(fig)

    # Figure 3: Sequence length efficiency
    sle_cols = "SLE_tokens_per_sec"
    fig, ax = plt.subplots(figsize=(8, 5))
    sle_vals = [summary_df[summary_df["tokenizer"] == t][sle_cols].values[0]
                if not summary_df[summary_df["tokenizer"] == t].empty else 0
                for t in tokenizers]
    bars = ax.bar(tokenizers, sle_vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, sle_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Figure 3: Sequence Length Efficiency (tokens/second)")
    ax.set_ylabel("Tokens per Second")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig3_sle_bar.png"), dpi=120)
    plt.close(fig)

    # Figure 4: CC preservation heatmap
    cc_cols = [c for c in summary_df.columns if c.startswith("CPR_CC")]
    if cc_cols:
        heatmap_data = summary_df[summary_df["tokenizer"].isin(tokenizers)].set_index("tokenizer")[cc_cols]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(heatmap_data.astype(float), annot=True, fmt=".2f", cmap="YlGn",
                    vmin=0, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title("Figure 4: CC Preservation Rate")
        ax.set_xticklabels([c.replace("CPR_", "") for c in cc_cols])
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "fig4_cc_heatmap.png"), dpi=120)
        plt.close(fig)

    # Figure 5: Vocabulary utilization
    vu_vals = [summary_df[summary_df["tokenizer"] == t]["VU"].values[0]
               if not summary_df[summary_df["tokenizer"] == t].empty else 0
               for t in tokenizers]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, tok, val, color in zip(axes, tokenizers, vu_vals, colors):
        used = val
        unused = max(0.0, 1.0 - val)
        ax.pie([used, unused], labels=["Used", "Unused"],
               colors=[color, "#ecf0f1"], autopct="%1.1f%%", startangle=90)
        ax.set_title(f"{tok}\n(VU={val:.2%})")
    fig.suptitle("Figure 5: Vocabulary Utilization")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "fig5_vocab_utilization.png"), dpi=120)
    plt.close(fig)

    print(f"Plots saved to {PLOTS_DIR}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    midi_files = sorted(glob.glob(os.path.join(MIDI_DIR, "*.mid")))
    if not midi_files:
        print(f"No MIDI files found in {MIDI_DIR}. Run generate_ambient_test_midi.py first.")
        return

    print(f"Evaluating {len(midi_files)} MIDI files...\n")

    all_results = {}
    all_vu = {}

    print("Evaluating remi_original...")
    per_file_orig, vu_orig = evaluate_remi_original(midi_files)
    all_results["remi_original"] = per_file_orig
    all_vu["remi_original"] = vu_orig
    print(f"  Done. {len(per_file_orig)} files processed.")

    print("Evaluating remi_fixed...")
    per_file_fixed, vu_fixed = evaluate_remi_fixed(midi_files)
    all_results["remi_fixed"] = per_file_fixed
    all_vu["remi_fixed"] = vu_fixed
    print(f"  Done. {len(per_file_fixed)} files processed.")

    print("Evaluating tsd_ambient...")
    per_file_tsd, vu_tsd = evaluate_tsd_ambient(midi_files)
    all_results["tsd_ambient"] = per_file_tsd
    all_vu["tsd_ambient"] = vu_tsd
    print(f"  Done. {len(per_file_tsd)} files processed.")

    # Build per-file dataframe
    rows = []
    for tok_name, file_rows in all_results.items():
        for row in file_rows:
            row = dict(row)
            row["tokenizer"] = tok_name
            rows.append(row)
    per_file_df = pd.DataFrame(rows)

    # Add VU per tokenizer
    for tok_name, vu_dict in all_vu.items():
        for col, val in vu_dict.items():
            per_file_df.loc[per_file_df["tokenizer"] == tok_name, col] = val

    per_file_df.to_csv(os.path.join(RESULTS_DIR, "per_file_metrics.csv"), index=False)
    print(f"\nPer-file metrics saved to {RESULTS_DIR}/per_file_metrics.csv")

    # Build summary
    numeric_cols = [c for c in per_file_df.columns if c not in ("file", "tokenizer")]
    summary = per_file_df.groupby("tokenizer")[numeric_cols].mean().reset_index()
    summary.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)
    print(f"Summary metrics saved to {RESULTS_DIR}/metrics_summary.csv\n")

    # Print summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    key_cols = ["RF_f1", "DQE_mean", "DQE_long_notes", "SLE_tokens_per_sec",
                "TRC", "VU", "CPR_CC91", "CPR_CC74", "CPR_CC64"]
    key_cols = [c for c in key_cols if c in summary.columns]
    print(summary[["tokenizer"] + key_cols].to_string(index=False, float_format="{:.3f}".format))

    plot_results(summary, per_file_df)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
