"""
Reproducibility test: compare datasets branch vs. main branch outputs for the same config.

Both branches run the same container with the same config and seeds. The datasets branch
refactored the generation code into a new module. This script verifies the outputs match.

Usage:
    python scripts/test_datasets_module.py \
        --config configs/neuromotion.json \
        --container /path/to/muniverse_neuromotion.sif \
        --engine singularity \
        --output data/reproducibility_test
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config_zero_noise(config_path: str) -> dict:
    with open(config_path) as f:
        config = json.load(f)
    config["RecordingConfiguration"]["NoiseLeveldb"] = None
    config["RecordingConfiguration"]["NoiseSeed"] = None
    return config


def load_datasets_results(output_dir: str) -> dict:
    """Load the single .npz written by generate_synthetic_recording."""
    path = os.path.join(output_dir, "emg_data.npz")
    return dict(np.load(path, allow_pickle=True))


def load_main_results(run_dir: str, muscle: str, subject_seed: int) -> dict:
    """
    Load the per-signal .npz files written by the main-branch _run_neuromotion.py.
    Filenames follow the pattern {subject_id}_{muscle}_{key}.npz where
    subject_id defaults to subject_{subject_seed} when SubjectID is absent from config.
    """
    import pdb; pdb.set_trace()
    subject_id = f"subject_{subject_seed}"
    prefix = f"{subject_id}_{muscle}"
    keys = {"emg": "emg", "spikes": "spikes", "muaps": "muaps"}
    results = {}
    for key, npz_key in keys.items():
        try:
            path = os.path.join(run_dir, f"{prefix}_{key}.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expected output not found: {path}")
            results[key] = np.load(path, allow_pickle=True)[npz_key]
        except:
            path = os.path.join(run_dir, f"{prefix}_{key}.npy")
            results[key] = np.load(path)
    return results


# ---------------------------------------------------------------------------
# Run: datasets branch
# ---------------------------------------------------------------------------

def run_datasets(config: dict, output_dir: str, engine: str, container: str) -> dict:
    """
    Generate a recording using the datasets branch (current branch).
    Imports directly since we are already on the datasets branch.
    Skips generation if emg_data.npz already exists.
    """
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "emg_data.npz")

    if os.path.exists(npz_path):
        print(f"[datasets] Found existing output at {npz_path}, skipping generation.")
        return load_datasets_results(output_dir)

    # Write modified config to a temp file so generate_synthetic_recording can read it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        tmp_config = f.name

    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        from muniverse import datasets  # noqa: datasets branch import

        results = datasets.generate_synthetic_recording(
            input_config=tmp_config,
            output_dir=output_dir,
            engine=engine,
            container=container,
            cache_dir=None,
            verbose=False,
        )
    finally:
        os.unlink(tmp_config)

    return results


# ---------------------------------------------------------------------------
# Run: main branch (via git worktree + subprocess)
# ---------------------------------------------------------------------------

def run_main_branch(config: dict, output_dir: str, engine: str, container: str) -> dict:
    """
    Generate a recording using the main branch via a temporary git worktree.
    Skips generation if the expected output files already exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    config["MovementConfiguration"]["MovementDOF"] = 'Radial-Ulnar-deviation'
    muscle = config["MovementConfiguration"]["TargetMuscle"]
    subject_seed = config["SubjectConfiguration"]["SubjectSeed"]
    subject_id = f"subject_{subject_seed}"
    emg_path = os.path.join(output_dir, f"{subject_id}_{muscle}_emg.npz")

    if os.path.exists(emg_path):
        print(f"[main] Found existing output at {output_dir}, skipping generation.")
        return load_main_results(output_dir, muscle, subject_seed)

    worktree_path = tempfile.mkdtemp(prefix="muniverse_main_")
    try:
        print(f"[main] Creating worktree for main branch at {worktree_path} ...")
        subprocess.run(
            ["git", "worktree", "add", "--force", worktree_path, "main"],
            cwd=REPO_ROOT, check=True,
        )

        # Main branch uses "EffortLevel" where the current config uses "TargetEffort".
        main_config = json.loads(json.dumps(config))
        params = main_config["MovementConfiguration"]["MovementProfileParameters"]
        if "EffortLevel" not in params and "TargetEffort" in params:
            params["EffortLevel"] = params["TargetEffort"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(main_config, f)
            tmp_config = f.name

        try:
            # Run main-branch generation inside the worktree via subprocess so that
            # its module imports resolve to the main branch's code, not datasets branch.
            run_script = os.path.join(REPO_ROOT, "scripts", "_run_main_generation.py")
            _write_main_generation_helper(run_script)

            result = subprocess.run(
                [sys.executable, run_script,
                 "--config", tmp_config,
                 "--output_dir", os.path.abspath(output_dir),
                 "--engine", engine,
                 "--container", container,
                 "--worktree", worktree_path],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"[main] STDOUT:\n{result.stdout}")
                print(f"[main] STDERR:\n{result.stderr}")
                raise RuntimeError("Main branch generation failed")
            print(result.stdout)
        finally:
            os.unlink(tmp_config)
            if os.path.exists(run_script):
                os.unlink(run_script)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", worktree_path],
            cwd=REPO_ROOT, check=False,
        )

    return load_main_results(output_dir, muscle, subject_seed)


def _write_main_generation_helper(path: str):
    """Write a small helper script that runs main-branch data_generation code."""
    script = '''\
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--output_dir")
parser.add_argument("--engine")
parser.add_argument("--container")
parser.add_argument("--worktree")
args = parser.parse_args()

sys.path.insert(0, os.path.join(args.worktree, "src"))
# Workaround: main branch calls logger.finalize(run_dir, engine, container) but
# BaseMetadataLogger.finalize only accepts (engine, container). Drop the extra arg.
import muniverse.utils.logging as _mlog
_orig_finalize = _mlog.BaseMetadataLogger.finalize
def _patched_finalize(self, *args, **kwargs):
    if len(args) == 3:   # (run_dir, engine, container) — strip run_dir
        args = args[1:]
    return _orig_finalize(self, *args, **kwargs)
_mlog.BaseMetadataLogger.finalize = _patched_finalize

from muniverse.data_generation import generate_recording

run_dir = generate_recording({
    "input_config": args.config,
    "output_dir": args.output_dir,
    "engine": args.engine,
    "container": args.container,
    "cache_dir": os.path.join(args.output_dir, "cache"),
})
print(f"[main] Output in: {run_dir}")
'''
    with open(path, "w") as f:
        f.write(script)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def get_git_hash(ref: str = "HEAD") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", ref],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except subprocess.CalledProcessError:
        return "unknown"


def write_report(datasets_results: dict, main_results: dict, output_dir: str,
                 datasets_hash: str, main_hash: str):
    emg_ds_norm = float(np.linalg.norm(datasets_results["emg"]))
    emg_mn_norm = float(np.linalg.norm(main_results["emg"]))
    muap_ds_norm = float(np.linalg.norm(datasets_results["muaps"]))
    muap_mn_norm = float(np.linalg.norm(main_results["muaps"]))

    lines = [
        "Reproducibility Report",
        "======================",
        f"Date: {subprocess.check_output(['date'], text=True).strip()}",
        f"datasets branch commit : {datasets_hash}",
        f"main branch commit     : {main_hash}",
        "",
        "EMG norms:",
        f"  datasets : {emg_ds_norm:.6f}",
        f"  main     : {emg_mn_norm:.6f}",
        f"  |diff|   : {abs(emg_ds_norm - emg_mn_norm):.2e}",
        "",
        "MUAP norms:",
        f"  datasets : {muap_ds_norm:.6f}",
        f"  main     : {muap_mn_norm:.6f}",
        f"  |diff|   : {abs(muap_ds_norm - muap_mn_norm):.2e}",
    ]

    report_path = os.path.join(output_dir, "reproducibility_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] Saved to {report_path}")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _pick_active_unit(spikes) -> int:
    """Return index of first motor unit with at least one spike."""
    for i, s in enumerate(spikes):
        if len(s) > 0:
            return i
    return 0


def plot_comparison(datasets_results: dict, main_results: dict, output_dir: str, fs: int):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Reproducibility: datasets branch vs. main branch", fontsize=13)

    # --- EMG ---
    emg_ds = datasets_results["emg"]
    emg_mn = main_results["emg"]

    def flatten_emg(emg):
        if emg.ndim == 3:
            ch_rows, ch_cols, t = emg.shape
            return emg.reshape(ch_rows * ch_cols, t)
        return emg.T if emg.shape[0] > emg.shape[1] else emg

    emg_ds_flat = flatten_emg(emg_ds)
    emg_mn_flat = flatten_emg(emg_mn)
    ch_idx = 0
    t_ds = np.arange(emg_ds_flat.shape[1]) / fs
    t_mn = np.arange(emg_mn_flat.shape[1]) / fs

    axes[0].plot(t_ds, emg_ds_flat[ch_idx], lw=0.6, label="datasets")
    axes[0].plot(t_mn, emg_mn_flat[ch_idx], lw=0.6, color="tab:orange", alpha=0.7, label="main")
    axes[0].set_title(f"EMG channel {ch_idx}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")

    # --- Spike trains ---
    spikes_ds = datasets_results["spikes"]
    spikes_mn = main_results["spikes"]

    def to_spike_list(spikes):
        if isinstance(spikes, np.ndarray) and spikes.dtype == object:
            return [spikes[i] for i in range(len(spikes))]
        return list(spikes)

    spikes_ds = to_spike_list(spikes_ds)
    spikes_mn = to_spike_list(spikes_mn)

    unit_idx = _pick_active_unit(spikes_ds)
    unit_idx_mn = _pick_active_unit(spikes_mn)
    axes[1].eventplot(spikes_ds[unit_idx], lineoffsets=0.0, linelengths=0.7, linewidths=0.8, label="datasets")
    axes[1].eventplot(spikes_mn[unit_idx_mn], lineoffsets=0.0, linelengths=0.7, linewidths=0.8, colors="tab:orange", alpha=0.7, label="main")
    axes[1].set_title(f"Spike train MU {unit_idx}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_yticks([])
    axes[1].legend(loc="upper right")

    # --- MUAPs ---
    muaps_ds = datasets_results["muaps"]  # (n_units, steps, ch_rows, ch_cols, win)
    muaps_mn = main_results["muaps"]

    mu_idx, step_idx = 0, 0
    muap_ds = muaps_ds[mu_idx, step_idx]
    muap_mn = muaps_mn[mu_idx, step_idx]
    muap_ds_wave = muap_ds.reshape(-1, muap_ds.shape[-1]).mean(axis=0)
    muap_mn_wave = muap_mn.reshape(-1, muap_mn.shape[-1]).mean(axis=0)

    axes[2].plot(muap_ds_wave, lw=0.8, label="datasets")
    axes[2].plot(muap_mn_wave, lw=0.8, color="tab:orange", alpha=0.7, label="main")
    axes[2].set_title(f"MUAP MU {mu_idx} step {step_idx}")
    axes[2].set_xlabel("Sample")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "reproducibility_comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"[plot] Saved to {plot_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test datasets module reproducibility.")
    parser.add_argument("--config", default="/rds/general/user/pm1222/home/muniverse/configs/neuromotion.json",
                        help="Path to neuromotion config JSON")
    parser.add_argument("--container", default="/rds/general/user/pm1222/home/muniverse/environment/muniverse_neuromotion.sif",
                        help="Container image name (Docker) or .sif path (Singularity)")
    parser.add_argument("--engine", default="singularity", choices=["docker", "singularity"])
    parser.add_argument("--output", default="/rds/general/user/pm1222/home/muniverse/data/output/reproducibility_test",
                        help="Root directory for all outputs")
    args = parser.parse_args()

    config = load_config_zero_noise(args.config)
    fs = config["RecordingConfiguration"]["SamplingFrequency"]

    datasets_output = os.path.join(args.output, "datasets_branch")
    main_output = os.path.join(args.output, "main_branch")

    datasets_hash = get_git_hash("HEAD")
    main_hash = get_git_hash("main")

    print("=== Step 1: datasets branch ===")
    datasets_results = run_datasets(config, datasets_output, args.engine, args.container)

    print("\n=== Step 2: main branch ===")
    main_results = run_main_branch(config, main_output, args.engine, args.container)

    print("\n=== Step 3: Report ===")
    write_report(datasets_results, main_results, args.output, datasets_hash, main_hash)

    print("\n=== Step 4: Plotting comparison ===")
    plot_comparison(datasets_results, main_results, args.output, fs)

    print("\nDone.")


if __name__ == "__main__":
    main()
