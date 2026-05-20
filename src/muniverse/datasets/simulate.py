import json
import os
import subprocess
import numpy as np
import tempfile

from ..utils.logging import SimulationLogger


def validate_config(config_content, verbose=False):
    """
    Validate the configuration dictionary to ensure all required parameters are present and valid.
    
    Args:
        config_content (dict): Configuration dictionary to validate
        verbose (bool, optional): If True, print validation success message. Defaults to False.
        
    Raises:
        ValueError: If configuration is invalid with specific error message
    """
    # Check required top-level sections
    required_sections = ["SubjectConfiguration", "MovementConfiguration", "RecordingConfiguration"]
    for section in required_sections:
        if section not in config_content:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate SubjectConfiguration
    subject_cfg = config_content["SubjectConfiguration"]
    required_subject_params = ["SubjectSeed", "FibreDensity", "MuscleLabels", "MuscleMotorUnitCounts"]
    for param in required_subject_params:
        if param not in subject_cfg:
            raise ValueError(f"Missing required parameter in SubjectConfiguration: {param}")
    
    # Validate subject seed
    if not isinstance(subject_cfg["SubjectSeed"], int) or subject_cfg["SubjectSeed"] < 0:
        raise ValueError("SubjectSeed must be a non-negative integer")
    
    # Validate fibre density
    if not isinstance(subject_cfg["FibreDensity"], (int, float)) or not (100 <= subject_cfg["FibreDensity"] <= 300):
        raise ValueError("FibreDensity must be a number between 100 and 300")
    
    # Validate muscle labels and counts match
    muscle_labels = subject_cfg["MuscleLabels"]
    muscle_counts = subject_cfg["MuscleMotorUnitCounts"]
    if len(muscle_labels) != len(muscle_counts):
        raise ValueError("MuscleLabels and MuscleMotorUnitCounts must have the same length")
    
    # Validate MovementConfiguration
    movement_cfg = config_content["MovementConfiguration"]
    required_movement_params = ["TargetMuscle", "MovementDOF", "MovementProfileParameters"]
    for param in required_movement_params:
        if param not in movement_cfg:
            raise ValueError(f"Missing required parameter in MovementConfiguration: {param}")
    
    # Warn if TargetMuscle is not in MuscleLabels (valid for hybrid pipeline where TargetMuscle
    # may be a placeholder like "Tibialis Anterior" that is replaced by a real muscle at runtime).
    if movement_cfg["TargetMuscle"] not in muscle_labels:
        import warnings
        warnings.warn(
            f"TargetMuscle '{movement_cfg['TargetMuscle']}' not found in MuscleLabels. "
            "This is expected for hybrid pipelines."
        )
    
    # Validate movement DOF
    valid_dofs = ["Flexion-Extension", "Radial-Ulnar-Deviation"]
    if movement_cfg["MovementDOF"] not in valid_dofs:
        raise ValueError(f"MovementDOF must be one of: {valid_dofs}")
    
    # Validate movement profile parameters
    profile_params = movement_cfg["MovementProfileParameters"]
    required_profile_params = ["MovementDuration", "TargetEffort"]
    for param in required_profile_params:
        if param not in profile_params:
            raise ValueError(f"Missing required parameter in MovementProfileParameters: {param}")
    
    # Validate movement duration
    if not isinstance(profile_params["MovementDuration"], (int, float)) or profile_params["MovementDuration"] <= 0:
        raise ValueError("MovementDuration must be a positive number")
    
    # Validate target effort
    if not isinstance(profile_params["TargetEffort"], (int, float)) or not (1 <= profile_params["TargetEffort"] <= 100):
        raise ValueError("TargetEffort must be a number between 1 and 100")
    
    # Validate RecordingConfiguration
    recording_cfg = config_content["RecordingConfiguration"]
    required_recording_params = ["SamplingFrequency", "ElectrodeConfiguration", "FilterProperties"]
    for param in required_recording_params:
        if param not in recording_cfg:
            raise ValueError(f"Missing required parameter in RecordingConfiguration: {param}")
    
    # Validate sampling frequency
    if not isinstance(recording_cfg["SamplingFrequency"], (int, float)) or recording_cfg["SamplingFrequency"] <= 0:
        raise ValueError("SamplingFrequency must be a positive number")
    
    # Validate filter properties
    filter_cfg = recording_cfg["FilterProperties"]
    required_filter_params = ["FilterType", "CutoffFrequency", "FilterOrder"]
    for param in required_filter_params:
        if param not in filter_cfg:
            raise ValueError(f"Missing required parameter in FilterProperties: {param}")
    if not isinstance(filter_cfg["CutoffFrequency"], (int, float)) or filter_cfg["CutoffFrequency"] <= 0:
        raise ValueError("CutoffFrequency must be a positive number")
    if not isinstance(filter_cfg["FilterOrder"], int) or filter_cfg["FilterOrder"] <= 0:
        raise ValueError("FilterOrder must be a positive integer")

    # Validate electrode configuration
    electrode_cfg = recording_cfg["ElectrodeConfiguration"]
    required_electrode_params = ["NElectrodes", "NRows", "NCols"]
    for param in required_electrode_params:
        if param not in electrode_cfg:
            raise ValueError(f"Missing required parameter in ElectrodeConfiguration: {param}")
    for param in ["NElectrodes", "NRows", "NCols"]:
        if not isinstance(electrode_cfg[param], int) or electrode_cfg[param] <= 0:
            raise ValueError(f"{param} must be a positive integer")
    if electrode_cfg["NElectrodes"] != electrode_cfg["NRows"] * electrode_cfg["NCols"]:
        raise ValueError("NElectrodes must equal NRows * NCols")
    desired_cols = electrode_cfg.get("DesiredNCols", electrode_cfg["NCols"])
    if not isinstance(desired_cols, int) or desired_cols <= 0 or desired_cols > electrode_cfg["NCols"]:
        raise ValueError("DesiredNCols must be a positive integer no greater than NCols")
    
    if verbose:
        print("[INFO] Configuration validation passed")


def generate_recording(
    config,
    effort_profile,
    angle_profile,
    engine,
    container,
    *,
    muaps=None,
    muap_angle_labels=None,
    logger=None,
    verbose=False,
):
    """
    Run a recording simulation inside the container.

    Handles both the neuromotion pipeline (MUAPs generated by BioMime inside the
    container) and the hybrid pipeline (caller-provided MUAPs).  Pass ``muaps``
    and ``muap_angle_labels`` to activate the hybrid path.

    Args:
        config (dict): Configuration dictionary.
        effort_profile (np.ndarray): Effort profile array.
        angle_profile (np.ndarray): Angle profile array.
        engine (str): Container engine ("docker" or "singularity").
        container (str): Image name (Docker) or .sif path (Singularity).
        muaps (np.ndarray, optional): MUAPs array for hybrid pipeline.
            Must have shape (n_units, n_angle_labels, ch_rows, ch_cols, n_timepoints).
        muap_angle_labels (np.ndarray, optional): Angle labels for each MUAP step.
            Required when ``muaps`` is provided.
        logger (SimulationLogger, optional): Logger instance.
        verbose (bool): Print extra error detail on failure.

    Returns:
        dict: Simulation outputs with keys:
            emg, spikes, firing_rates, effort_profile, angle_profile,
            muaps, muap_angle_labels, properties, config
    """
    is_hybrid = muaps is not None

    if is_hybrid and muap_angle_labels is None:
        raise ValueError("'muap_angle_labels' is required when 'muaps' is provided")

    if logger is None:
        logger = SimulationLogger()
    logger.set_config(config)

    run_dir = tempfile.mkdtemp()
    try:
        # Write config
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Write input arrays
        input_arrays = dict(
            effort_profile=np.asarray(effort_profile),
            angle_profile=np.asarray(angle_profile),
        )

        if is_hybrid:
            muaps_array = np.asarray(muaps)
            muap_angle_labels_array = np.asarray(muap_angle_labels)

            if muaps_array.ndim != 5:
                raise ValueError(
                    f"muaps must have shape (n_units, n_angle_labels, ch_rows, ch_cols, n_timepoints), "
                    f"got {muaps_array.shape}"
                )
            if muaps_array.shape[1] != len(muap_angle_labels_array):
                raise ValueError(
                    f"muaps second dimension ({muaps_array.shape[1]}) must match "
                    f"muap_angle_labels length ({len(muap_angle_labels_array)})"
                )
            input_arrays["muaps"] = muaps_array
            input_arrays["muap_angle_labels"] = muap_angle_labels_array

        np.savez(os.path.join(run_dir, "input_data.npz"), **input_arrays)

        # Locate scripts
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "_run_neuromotion.py")
        run_script_path = os.path.join(current_dir, "_generate_recording.sh")

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")
        if not os.path.exists(run_script_path):
            raise FileNotFoundError(f"Shell script not found: {run_script_path}")

        # Run container
        cmd = [run_script_path, engine, container, script_path, run_dir]
        try:
            subprocess.run(cmd, check=True, cwd=current_dir, capture_output=verbose, text=verbose)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Recording generation failed: {e}")
            if verbose:
                if e.stdout:
                    print(f"[ERROR] stdout:\n{e.stdout}")
                if e.stderr:
                    print(f"[ERROR] stderr:\n{e.stderr}")
            logger.set_return_code("run.sh", e.returncode)
            logger.finalize(engine, container)
            raise

        logger.set_return_code("run.sh", 0)
        logger.finalize(engine, container)

        results = dict(np.load(os.path.join(run_dir, "emg_data.npz"), allow_pickle=True))
        print(f"[INFO] Recording generated successfully")
        return results

    finally:
        # Clean up temp dir even on failure (logger has already been finalised above)
        if os.path.exists(run_dir):
            import shutil
            shutil.rmtree(run_dir, ignore_errors=True)

