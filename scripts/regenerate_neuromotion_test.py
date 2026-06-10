"""
regenerate_neuromotion_test.py

Loads each trial from the Neuromotion-Test dataset, translates its simulation
config to the current datasets-module format, regenerates the data, and
produces comparison plots to assess reproducibility.

Usage:
    python scripts/regenerate_neuromotion_test.py \\
        --old_data /path/to/Neuromotion-Test/sub-sim02_ses-02_ECRB \\
        --container /path/to/muniverse_neuromotion.sif \\
        --engine singularity \\
        --output data/neuromotion_regen

Optional flags:
    --skip_regen   Skip regeneration; only plot from already-saved .npz files.
    --no_noise     Zero out NoiseLeveldb/NoiseSeed before regenerating.
"""

import argparse
import copy
import json
import os
import sys
import tempfile
from importlib.resources import files

import edfio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


# ---------------------------------------------------------------------------
# Config translation
# ---------------------------------------------------------------------------

def translate_config(old_config: dict) -> dict:
    """
    Build a new-format config by starting from the neuromotion.json default and
    patching in trial-specific values from old_config.

    Key conversions from old format:
    - EffortLevel → TargetEffort
    - MovementDOF case normalised ("Radial-Ulnar-deviation" → "Radial-Ulnar-Deviation")
    - Sinusoid angle: old TargetAngle (centre) + SinAmplitude (signed half-range)
      → new InitialAngle = centre + amplitude, TargetAngle = centre - amplitude
    - Triangular angle: InitialAngle defaults to 0
    """
    #default_path = os.path.join(REPO_ROOT, "configs", "neuromotion.json")
    default_path = str(files("muniverse").joinpath("configs/neuromotion.json"))
    with open(default_path) as f:
        config = json.load(f)

    old_subj = old_config["SubjectConfiguration"]
    old_mov = old_config["MovementConfiguration"]
    old_rec = old_config["RecordingConfiguration"]
    old_params = old_mov["MovementProfileParameters"]

    # --- SubjectConfiguration ---
    config["SubjectConfiguration"]["SubjectSeed"] = old_subj["SubjectSeed"]
    config["SubjectConfiguration"]["FibreDensity"] = old_subj["FibreDensity"]
    config["SubjectConfiguration"]["MuscleLabels"] = old_subj["MuscleLabels"]
    config["SubjectConfiguration"]["MuscleMotorUnitCounts"] = old_subj["MuscleMotorUnitCounts"]

    # --- MovementConfiguration ---
    config["MovementConfiguration"]["TargetMuscle"] = old_mov["TargetMuscle"]
    dof = old_mov.get("MovementDOF", "")
    if dof.lower() == "radial-ulnar-deviation":
        dof = "Radial-Ulnar-Deviation"
    config["MovementConfiguration"]["MovementDOF"] = dof

    new_params = config["MovementConfiguration"]["MovementProfileParameters"]
    new_params["EffortProfile"] = old_params["EffortProfile"]
    new_params["TargetEffort"] = old_params.get("TargetEffort", old_params.get("EffortLevel"))
    new_params["AngleProfile"] = old_params["AngleProfile"]
    new_params["NRepetitions"] = old_params.get("NRepetitions", 1)
    new_params["RestDuration"] = old_params["RestDuration"]
    new_params["RampDuration"] = old_params["RampDuration"]
    new_params["HoldDuration"] = old_params["HoldDuration"]
    new_params["SinFrequency"] = old_params.get("SinFrequency", 0)
    new_params["MovementDuration"] = old_params["MovementDuration"]

    angle_profile = old_params["AngleProfile"]
    target_angle_old = old_params.get("TargetAngle", 0)
    sin_amplitude = old_params.get("SinAmplitude", 0)

    if angle_profile == "Sinusoid":
        # Old: TargetAngle = centre, SinAmplitude = signed half-range.
        # New: offset = (target + initial)/2, amplitude = (target - initial)/2
        # So: initial = centre + sin_amplitude, target = centre - sin_amplitude
        new_params["InitialAngle"] = target_angle_old + sin_amplitude
        new_params["TargetAngle"] = target_angle_old - sin_amplitude
    elif angle_profile == "Triangular":
        new_params["InitialAngle"] = old_params.get("InitialAngle", 0)
        new_params["TargetAngle"] = target_angle_old
    else:
        new_params["InitialAngle"] = old_params.get("InitialAngle", 0)
        new_params["TargetAngle"] = target_angle_old

    # --- RecordingConfiguration ---
    config["RecordingConfiguration"]["SamplingFrequency"] = old_rec["SamplingFrequency"]
    config["RecordingConfiguration"]["NoiseSeed"] = old_rec.get("NoiseSeed")
    config["RecordingConfiguration"]["NoiseLeveldb"] = old_rec.get("NoiseLeveldb")
    if "ElectrodeConfiguration" in old_rec:
        old_elec = old_rec["ElectrodeConfiguration"]
        new_elec = config["RecordingConfiguration"]["ElectrodeConfiguration"]
        n_rows = old_elec.get("NRows", 10)
        # BioMime always outputs 32 columns; DesiredNCols is what the old config called NCols
        desired_n_cols = old_elec.get("DesiredNCols", old_elec.get("NCols", 10))
        new_elec["NRows"] = n_rows
        new_elec["NCols"] = 32
        new_elec["NElectrodes"] = n_rows * 32
        new_elec["DesiredNCols"] = desired_n_cols
        if "InterElectrodeDistance" in old_elec:
            new_elec["InterElectrodeDistance"] = old_elec["InterElectrodeDistance"]
    if "FilterProperties" in old_rec:
        old_filt = old_rec["FilterProperties"]
        new_filt = config["RecordingConfiguration"]["FilterProperties"]
        for key in ("FilterType", "CutoffFrequency", "FilterOrder"):
            if key in old_filt:
                new_filt[key] = old_filt[key]

    return config


def summarise_translation(old_cfg: dict, new_cfg: dict) -> list[str]:
    """Return a list of human-readable lines describing config changes."""
    lines = []
    old_params = old_cfg["MovementConfiguration"]["MovementProfileParameters"]
    new_params = new_cfg["MovementConfiguration"]["MovementProfileParameters"]

    old_dof = old_cfg["MovementConfiguration"]["MovementDOF"]
    new_dof = new_cfg["MovementConfiguration"]["MovementDOF"]
    if old_dof != new_dof:
        lines.append(f"  MovementDOF: '{old_dof}' → '{new_dof}'")

    if "EffortLevel" in old_params:
        lines.append(f"  EffortLevel={old_params['EffortLevel']} → TargetEffort={new_params['TargetEffort']}")

    old_angle = old_params.get("AngleProfile", "Constant")
    if old_angle == "Sinusoid":
        lines.append(
            f"  Sinusoid angle: TargetAngle={old_params['TargetAngle']}, "
            f"SinAmplitude={old_params.get('SinAmplitude', 0)} → "
            f"InitialAngle={new_params['InitialAngle']}, TargetAngle={new_params['TargetAngle']}"
        )
    elif old_angle == "Triangular":
        lines.append(f"  Triangular angle: added InitialAngle={new_params.get('InitialAngle', 0)}")

    return lines or ["  (no structural changes)"]


# ---------------------------------------------------------------------------
# Load old data
# ---------------------------------------------------------------------------

def load_old_emg(edf_path: str, n_rows: int = 10) -> np.ndarray:
    """
    Read EDF and return emg of shape (n_rows, n_cols, n_samples).
    The number of columns is inferred from total signal count.
    """
    edf = edfio.read_edf(edf_path)
    signals = np.stack([s.data for s in edf.signals], axis=0)  # (n_ch, n_samples)
    n_ch, n_samples = signals.shape
    n_cols = n_ch // n_rows
    return signals.reshape(n_rows, n_cols, n_samples)


def load_old_spikes(tsv_path: str) -> list:
    """
    Read spikes TSV and return a list of spike-time arrays, one per motor unit.
    source_id is the MU index; spike_time is in samples.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    n_units = int(df["source_id"].max()) + 1
    spikes = [np.array(df[df["source_id"] == i]["spike_time"].values, dtype=int)
              for i in range(n_units)]
    return spikes


def load_old_effort_angle(config: dict, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive old effort and angle profiles from the old config params."""
    from muniverse.datasets.movement import (
        generate_effort_profile, generate_angle_profile,
    )
    # Deep copy to prevent validate_effort_profile_config from mutating the caller's config
    cfg = copy.deepcopy(config)
    effort, _ = generate_effort_profile(cfg)
    angle, _ = generate_angle_profile(cfg)
    return effort, angle


# ---------------------------------------------------------------------------
# Regenerate
# ---------------------------------------------------------------------------

def regenerate_trial(new_config: dict, output_path: str, engine: str, container: str) -> dict:
    """
    Generate a recording with the current datasets module and cache result.
    Returns the result dict (emg, spikes, effort_profile, angle_profile, ...).
    """
    if os.path.exists(output_path):
        print(f"    [cache] Loading existing {output_path}")
        return dict(np.load(output_path, allow_pickle=True))

    from muniverse.datasets import generate_synthetic_recording

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(new_config, f)
        tmp_cfg = f.name

    out_dir = os.path.dirname(output_path)
    try:
        results = generate_synthetic_recording(
            input_config=tmp_cfg,
            output_dir=out_dir,
            engine=engine,
            container=container,
            cache_dir=None,
            verbose=True,
        )
        # generate_synthetic_recording saves to out_dir/emg_data.npz; rename to our path.
        default_path = os.path.join(out_dir, "emg_data.npz")
        if default_path != output_path and os.path.exists(default_path):
            os.rename(default_path, output_path)
        else:
            np.savez_compressed(output_path, **results)
    finally:
        os.unlink(tmp_cfg)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _first_active(spikes: list) -> int:
    for i, s in enumerate(spikes):
        if len(s) > 0:
            return i
    return 0


def plot_trial_comparison(
    trial_name: str,
    old_emg: np.ndarray,
    new_results: dict,
    old_spikes: list,
    old_effort: np.ndarray,
    old_angle: np.ndarray,
    new_config: dict,
    fs: float,
    save_path: str,
):
    """
    Four-panel comparison: effort profile, angle profile, spike raster, EMG channel.
    """
    new_emg = new_results["emg"]       # (n_rows, n_cols, T) or (n_ch, T)
    new_spikes_raw = new_results["spikes"]
    new_effort = new_results.get("effort_profile", np.array([]))
    new_angle = new_results.get("angle_profile", np.array([]))

    # Normalise spike list
    def _to_list(s):
        if isinstance(s, np.ndarray) and s.dtype == object:
            return [s[i] for i in range(len(s))]
        return list(s)

    new_spikes = _to_list(new_spikes_raw)

    # Flatten EMG to (n_ch, T) for easy channel access
    def _flatten(emg):
        if emg.ndim == 3:
            nr, nc, t = emg.shape
            return emg.reshape(nr * nc, t)
        return emg if emg.shape[0] < emg.shape[1] else emg.T

    old_emg_flat = _flatten(old_emg)
    new_emg_flat = _flatten(new_emg)

    t_old = np.arange(old_emg_flat.shape[1]) / fs
    t_new = np.arange(new_emg_flat.shape[1]) / fs
    t_eff_old = np.arange(len(old_effort)) / fs
    t_eff_new = np.arange(len(new_effort)) / fs
    t_ang_old = np.arange(len(old_angle)) / fs
    t_ang_new = np.arange(len(new_angle)) / fs

    fig, axes = plt.subplots(5, 1, figsize=(13, 15))
    fig.suptitle(f"Reproducibility: {trial_name}", fontsize=11, fontweight="bold")

    # 1. Effort profile
    ax = axes[0]
    ax.plot(t_eff_old, old_effort * 100, lw=1.2, label="old data (from config)")
    ax.plot(t_eff_new, new_effort * 100, lw=1.2, ls="--", label="regen")
    ax.set_ylabel("Effort (%MVC)")
    ax.set_title("Effort profile")
    ax.legend(loc="upper right", fontsize=8)

    # 2. Angle profile
    ax = axes[1]
    ax.plot(t_ang_old, old_angle, lw=1.2, label="old data (from config)")
    ax.plot(t_ang_new, new_angle, lw=1.2, ls="--", label="regen")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Angle profile")
    ax.legend(loc="upper right", fontsize=8)

    # 3. Spike count by motor unit
    ax = axes[2]
    old_counts = np.array([len(s) for s in old_spikes])
    new_counts = np.array([len(s) for s in new_spikes])
    mu_idx_old = np.arange(len(old_counts))
    mu_idx_new = np.arange(len(new_counts))
    ax.plot(mu_idx_old, old_counts, lw=1.2, label=f"old  (total={old_counts.sum()})")
    ax.plot(mu_idx_new, new_counts, lw=1.2, ls="--", color="tab:orange",
            label=f"regen (total={new_counts.sum()})")
    ax.set_ylabel("Spike count")
    ax.set_xlabel("Motor unit index")
    ax.set_title("Spike count by motor unit")
    ax.legend(loc="upper right", fontsize=8)

    # 4. Spike raster (first active unit from each)
    ax = axes[3]
    mu_old = _first_active(old_spikes)
    mu_new = _first_active(new_spikes)
    if len(old_spikes[mu_old]) > 0:
        ax.eventplot(old_spikes[mu_old] / fs, lineoffsets=1.0, linelengths=0.7,
                     linewidths=0.8, label=f"old MU{mu_old}")
    if len(new_spikes) > mu_new and len(new_spikes[mu_new]) > 0:
        ax.eventplot(np.array(new_spikes[mu_new]) / fs, lineoffsets=0.0, linelengths=0.7,
                     linewidths=0.8, colors="tab:orange", label=f"regen MU{mu_new}")
    ax.set_title("Spike trains (first active MU)")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["regen", "old"])
    ax.legend(loc="upper right", fontsize=8)

    # 5. EMG channel 0
    ax = axes[4]
    ax.plot(t_old, old_emg_flat[0], lw=0.5, label="old ch0")
    ax.plot(t_new, new_emg_flat[0], lw=0.5, alpha=0.8, color="tab:orange", label="regen ch0")
    ax.set_ylabel("Amplitude (V)")
    ax.set_xlabel("Time (s)")
    ax.set_title("EMG channel 0")
    ax.legend(loc="upper right", fontsize=8)

    # Stats annotation
    min_t = min(old_emg_flat.shape[1], new_emg_flat.shape[1])
    corr = float(np.corrcoef(old_emg_flat[0, :min_t], new_emg_flat[0, :min_t])[0, 1])
    rms_old = float(np.sqrt(np.mean(old_emg_flat[0, :min_t] ** 2)))
    rms_new = float(np.sqrt(np.mean(new_emg_flat[0, :min_t] ** 2)))
    fig.text(
        0.99, 0.01,
        f"ch0 corr={corr:.3f}  RMS old={rms_old:.4f}  regen={rms_new:.4f}",
        ha="right", va="bottom", fontsize=8, color="gray",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    [plot] Saved {save_path}")
    return corr, rms_old, rms_new


def reconstruct_full_emg(muaps, spikes_raw, muap_angle_labels, angle_profile):
    """Reconstruct the full (n_rows, n_cols, T) EMG from the cached MUAP library."""
    if isinstance(spikes_raw, np.ndarray) and spikes_raw.dtype == object:
        spikes = [spikes_raw[i] for i in range(len(spikes_raw))]
    else:
        spikes = list(spikes_raw)

    n_units, steps, n_rows, n_cols, win = muaps.shape
    T = len(angle_profile)
    offset = win // 2
    full_emg = np.zeros((n_rows, n_cols, T))

    for unit in range(n_units):
        for firing in spikes[unit]:
            if int(firing) >= T:
                continue
            curr_angle = angle_profile[int(firing)]
            muap_idx = int(np.argmin(np.abs(muap_angle_labels - curr_angle)))
            curr_muap = muaps[unit, muap_idx]
            init_emg = max(0, firing - offset)
            end_emg = min(firing + offset, T)
            init_muap = init_emg - (firing - offset)
            end_muap = end_emg - (firing + offset) + win
            full_emg[:, :, init_emg:end_emg] += curr_muap[:, :, init_muap:end_muap]

    return full_emg


def diagnose_electrode_selection(new_results, desired_cols=10):
    """
    Tests the electrode selection bug hypothesis.

    Reconstructs the full 10×32 EMG from cached muaps/spikes and applies both:
      - NEW code's correct selection: (n_rows, desired_cols, T)
      - OLD code's BUGGY selection: col-major indices on a row-major array

    Prints RMS of each selection so we can check if the bug explains the ~3.3x amplitude gap.
    """
    muaps = new_results['muaps']            # (n_units, steps, 10, 32, win)
    spikes_raw = new_results['spikes']
    muap_angle_labels = new_results['muap_angle_labels']
    angle_profile = new_results['angle_profile']

    print("  [elec_diag] Reconstructing full 10×32 EMG from cached muaps…")
    full_emg = reconstruct_full_emg(muaps, spikes_raw, muap_angle_labels, angle_profile)
    # full_emg: (10, 32, T)
    n_rows, n_cols, T = full_emg.shape

    # --- New code: correct row-major selection ---
    rms_per_col = np.sqrt(np.mean(full_emg ** 2, axis=(0, 2)))  # (32,)
    center_col = int(np.argmax(rms_per_col))
    half = desired_cols // 2
    selected_cols = [(center_col - half + i) % n_cols for i in range(desired_cols)]
    new_sel_emg = full_emg[:, selected_cols, :]   # (10, 10, T)
    rms_new_correct = float(np.sqrt(np.mean(new_sel_emg ** 2)))
    rms_new_ch0 = float(np.sqrt(np.mean(new_sel_emg[0, 0] ** 2)))

    # --- Old code: BUGGY col-major indices applied to row-major (T, 320) array ---
    # channel i in (T, 320) = electrode (row_{i//32}, col_{i%32})
    emg_flat = full_emg.reshape(n_rows * n_cols, T).T   # (T, 320) row-major
    old_indices = [col * n_rows + row for col in selected_cols for row in range(n_rows)]
    old_sel_emg = emg_flat[:, old_indices]              # (T, 100)
    rms_old_buggy = float(np.sqrt(np.mean(old_sel_emg ** 2)))
    rms_old_buggy_ch0 = float(np.sqrt(np.mean(old_sel_emg[:, 0] ** 2)))

    # What physical electrode does old ch0 map to?
    old_ch0_idx = old_indices[0]
    old_ch0_phys_row = old_ch0_idx // n_cols
    old_ch0_phys_col = old_ch0_idx % n_cols

    print(f"  [elec_diag] center_col = {center_col},  selected_cols = {selected_cols}")
    print(f"  [elec_diag] old ch0 index in (T,320) = {old_ch0_idx}"
          f"  -> physical (row={old_ch0_phys_row}, col={old_ch0_phys_col})")
    print(f"  [elec_diag] new correct selection  -- all-ch RMS: {rms_new_correct:.6f}  ch0 RMS: {rms_new_ch0:.6f}")
    print(f"  [elec_diag] old BUGGY selection    -- all-ch RMS: {rms_old_buggy:.6f}  ch0 RMS: {rms_old_buggy_ch0:.6f}")
    print(f"  [elec_diag] RMS ratio (new/old)    -- all-ch: {rms_new_correct/rms_old_buggy:.2f}x  "
          f"ch0: {rms_new_ch0/rms_old_buggy_ch0:.2f}x")

    # Per-channel RMS arrays for plotting
    rms_new_per_ch = np.sqrt(np.mean(new_sel_emg.reshape(-1, T) ** 2, axis=1))  # (100,)
    rms_buggy_per_ch = np.sqrt(np.mean(old_sel_emg.T ** 2, axis=1))             # (100,)

    return center_col, rms_new_per_ch, rms_buggy_per_ch


def plot_channel_rms_comparison(
    old_emg: np.ndarray,
    rms_new_per_ch: np.ndarray,
    rms_buggy_per_ch: np.ndarray,
    trial_name: str,
    save_path: str,
):
    """
    Plot per-channel RMS for old EDF data, new correct selection, and new buggy selection.
    old_emg: (n_rows, n_cols, T) loaded from EDF.
    """
    T = old_emg.shape[2]
    rms_old_per_ch = np.sqrt(np.mean(old_emg.reshape(-1, T) ** 2, axis=1))  # (100,)

    ch_idx = np.arange(len(rms_old_per_ch))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Per-channel RMS: {trial_name}", fontsize=9, fontweight="bold")

    # Line plot
    ax = axes[0]
    ax.plot(ch_idx, rms_old_per_ch,   lw=1.2, label="old EDF (buggy selection saved to disk)")
    ax.plot(ch_idx, rms_new_per_ch,   lw=1.2, label="new (correct column selection)")
    ax.plot(ch_idx, rms_buggy_per_ch, lw=1.2, ls="--", label="new EMG with old buggy selection")
    ax.set_xlabel("Channel index (flattened)")
    ax.set_ylabel("RMS amplitude")
    ax.set_title("Per-channel RMS by channel order")
    ax.legend(fontsize=8)

    # Histogram
    ax = axes[1]
    all_vals = np.concatenate([rms_old_per_ch, rms_new_per_ch, rms_buggy_per_ch])
    bins = np.linspace(0, all_vals.max() * 1.05, 30)
    ax.hist(rms_old_per_ch,   bins=bins, alpha=0.6, label="old EDF")
    ax.hist(rms_new_per_ch,   bins=bins, alpha=0.6, label="new (correct)")
    ax.hist(rms_buggy_per_ch, bins=bins, alpha=0.6, label="new EMG, old buggy sel.")
    ax.set_xlabel("RMS amplitude")
    ax.set_ylabel("Channel count")
    ax.set_title("RMS distribution across channels")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"    [plot] Saved {save_path}")


def plot_summary(summary_rows: list, output_dir: str):
    """Bar chart summarising per-trial channel-0 correlation."""
    if not summary_rows:
        return
    names = [r["name"] for r in summary_rows]
    corrs = [r["corr"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 4))
    bars = ax.bar(range(len(names)), corrs, color="steelblue")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Pearson r (ch0, old vs regen)")
    ax.set_title("EMG reproducibility per trial")
    ax.set_ylim([-1.05, 1.1])
    for bar, val in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    path = os.path.join(output_dir, "summary_reproducibility.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[summary] Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_trials(old_data_dir: str) -> list[dict]:
    """Return list of dicts with paths for each trial, searching recursively."""
    trials = []
    for root, _, files in os.walk(old_data_dir):
        for fname in sorted(files):
            if not fname.endswith("_simulation.json"):
                continue
            base = fname.replace("_simulation.json", "")
            trial = {
                "name": base,
                "sim_json": os.path.join(root, fname),
                "edf": os.path.join(root, f"{base}_emg.edf"),
                "tsv": os.path.join(root, f"{base}_spikes.tsv"),
            }
            if os.path.exists(trial["edf"]) and os.path.exists(trial["tsv"]):
                trials.append(trial)
            else:
                print(f"[warn] Skipping {base}: missing EDF or TSV")
    trials.sort(key=lambda t: t["name"])
    return trials


def main():
    parser = argparse.ArgumentParser(description="Regenerate Neuromotion-Test trials and compare to old data.")
    parser.add_argument("--old_data", required=True,
                        help="Path to old data directory (sub-sim02_ses-02_ECRB)")
    parser.add_argument("--container", default=None,
                        help="Container image name (Docker) or .sif path (Singularity)")
    parser.add_argument("--engine", default="singularity", choices=["docker", "singularity"])
    parser.add_argument("--output", default="data/neuromotion_regen",
                        help="Output directory for regenerated data and plots")
    parser.add_argument("--skip_regen", action="store_true",
                        help="Skip regeneration; only plot from cached .npz files")
    parser.add_argument("--no_noise", action="store_true",
                        help="Set NoiseLeveldb and NoiseSeed to None before regenerating")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    trials = discover_trials(os.path.abspath(args.old_data))
    print(f"Found {len(trials)} trials in {args.old_data}\n")

    summary_rows = []

    for trial in trials:
        name = trial["name"]
        print(f"=== {name} ===")

        # Load old simulation config
        with open(trial["sim_json"]) as f:
            sim_json = json.load(f)
        old_config = sim_json["InputData"]["Configuration"]

        # Translate to new format
        new_config = translate_config(old_config)

        if args.no_noise:
            new_config["RecordingConfiguration"]["NoiseLeveldb"] = None
            new_config["RecordingConfiguration"]["NoiseSeed"] = None

        print("  Config translation:")
        for line in summarise_translation(old_config, new_config):
            print(line)

        fs = new_config["RecordingConfiguration"]["SamplingFrequency"]

        # Load old data
        print("  Loading old data...")
        old_emg = load_old_emg(trial["edf"])
        old_spikes = load_old_spikes(trial["tsv"])
        old_effort, old_angle = load_old_effort_angle(new_config, fs)

        # Regenerate
        trial_output_dir = os.path.join(output_dir, name)
        os.makedirs(trial_output_dir, exist_ok=True)
        npz_path = os.path.join(trial_output_dir, "regen_emg_data.npz")

        if args.skip_regen:
            if not os.path.exists(npz_path):
                print(f"  [skip_regen] No cached result at {npz_path}, skipping trial.")
                continue
            print("  Loading cached regenerated data...")
            new_results = dict(np.load(npz_path, allow_pickle=True))
        else:
            if args.container is None:
                print("  [error] --container is required unless --skip_regen is set.")
                sys.exit(1)
            print("  Regenerating...")
            new_results = regenerate_trial(new_config, npz_path, args.engine, args.container)

        # Electrode selection bug diagnostic
        elec_cfg = new_config["RecordingConfiguration"]["ElectrodeConfiguration"]
        desired_cols = elec_cfg.get("DesiredNCols", elec_cfg["NCols"])
        _, rms_new_per_ch, rms_buggy_per_ch = diagnose_electrode_selection(new_results, desired_cols=desired_cols)
        rms_plot_path = os.path.join(trial_output_dir, "channel_rms_comparison.png")
        plot_channel_rms_comparison(old_emg, rms_new_per_ch, rms_buggy_per_ch, name, rms_plot_path)

        # Save translated config next to regen result for reference
        cfg_path = os.path.join(trial_output_dir, "translated_config.json")
        if not os.path.exists(cfg_path):
            with open(cfg_path, "w") as f:
                json.dump(new_config, f, indent=2)

        # Plot comparison
        plot_path = os.path.join(trial_output_dir, "comparison.png")
        corr, rms_old, rms_new = plot_trial_comparison(
            trial_name=name,
            old_emg=old_emg,
            new_results=new_results,
            old_spikes=old_spikes,
            old_effort=old_effort,
            old_angle=old_angle,
            new_config=new_config,
            fs=fs,
            save_path=plot_path,
        )
        summary_rows.append({"name": name, "corr": corr, "rms_old": rms_old, "rms_new": rms_new})
        print()

    # Summary plot
    plot_summary(summary_rows, output_dir)

    # Summary CSV
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, "summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"[summary] Saved {csv_path}")
        for _, row in df.iterrows():
            print(f"  {row['name']}")
            print(f"    corr={row['corr']:.4f}  rms_old={row['rms_old']:.6f}  rms_new={row['rms_new']:.6f}")


if __name__ == "__main__":
    main()
