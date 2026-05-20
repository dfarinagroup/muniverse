import numpy as np
from typing import Dict, Tuple
import warnings


def generate_effort_profile(config: Dict) -> np.ndarray:
    """
    Generate effort profile from config.
    
    Args:
        config: Configuration dict with MovementConfiguration and RecordingConfiguration
        
    Returns:
        Effort profile array
    """
    fs = config.get("RecordingConfiguration").get("SamplingFrequency")
    params = config.get("MovementConfiguration").get("MovementProfileParameters")
    
    params = validate_effort_profile_config(params)
    duration = params.get("MovementDuration")
    effort_profile = _create_effort_profile(params, int(duration * fs), fs)
    return effort_profile, params


def generate_angle_profile(config: Dict) -> np.ndarray:
    """
    Generate angle profile from config.
    
    Args:
        config: Configuration dict with MovementConfiguration and RecordingConfiguration
        
    Returns:
        Angle profile array
    """
    fs = config.get("RecordingConfiguration").get("SamplingFrequency")
    params = config.get("MovementConfiguration").get("MovementProfileParameters")
    movement_dof = config.get("MovementConfiguration").get("MovementDOF")

    params = validate_angle_profile_config(params)
    duration = params.get("MovementDuration")
    angle_profile = _create_angle_profile(params, int(duration * fs), fs, movement_dof)
    
    return angle_profile, params


def validate_effort_profile_config(params: Dict) -> Dict:
    """Validate movement configuration."""
    # Force default values
    if params.get("EffortProfile") == "Trapezoid":
        params["InitialEffort"] = 0
        params["SinFrequency"] = 0
    elif params.get("EffortProfile") == "Triangular":
        params["InitialEffort"] = 0
        params["HoldDuration"] = 0
        params["SinFrequency"] = 0
    elif params.get("EffortProfile") == "Sinusoid":
        pass
    elif params.get("EffortProfile") == "Ballistic":
        params["InitialEffort"] = 0
        if params.get("NRepetitions") < 5:
            warnings.warn(f"NRepetitions is less than 5. Setting to 5.")
            params["NRepetitions"] = 5
        params["HoldDuration"] = 0
        params["SinFrequency"] = 0
    elif params.get("EffortProfile") == "Constant":
        params["HoldDuration"] = 0
        params["SinFrequency"] = 0
    else:
        raise ValueError(f"Unsupported effort profile type: '{params.get('EffortProfile')}'")

    params["MovementDuration"] = _compute_movement_duration(params)
    return params


def validate_angle_profile_config(params: Dict) -> Dict:
    """Validate angle profile configuration."""
    # Force default values
    
    return params


def _compute_movement_duration(params: Dict) -> float:
    """Compute movement duration."""
    rest_duration = params.get("RestDuration")
    ramp_duration = params.get("RampDuration")
    hold_duration = params.get("HoldDuration")
    n_reps = params.get("NRepetitions", 1)
    # Each cycle has a leading AND trailing rest segment, so rest contributes 2x.
    if params.get("EffortProfile") == "Ballistic":
        return n_reps * (2 * rest_duration + ramp_duration)
    else:
        return (2 * rest_duration + 2 * ramp_duration + hold_duration) * n_reps


def _create_effort_profile(params: Dict, samples: int, fs: float) -> np.ndarray:
    """Create effort profile based on parameters."""
    target_effort = params.get("TargetEffort") / 100.0
    profile_type = params.get("EffortProfile", "Trapezoid")

    # Read parameters from config
    rest_duration = params.get("RestDuration")
    ramp_duration = params.get("RampDuration")
    hold_duration = params.get("HoldDuration")
    n_reps = params.get("NRepetitions", 1)
    duration = _compute_movement_duration(params)
    samples = int(duration * fs)

    if profile_type == "Trapezoid":
        ramp_duration = params.get("RampDuration")
        hold_duration = params.get("HoldDuration")
        return _trapezoid_profile(rest_duration, ramp_duration, hold_duration, target_effort, n_reps, samples, fs)
    elif profile_type == "Triangular":
        ramp_duration = params.get("RampDuration")
        return _triangular_profile(rest_duration, ramp_duration, n_reps, target_effort, samples, fs)
    elif profile_type == "Sinusoid":
        sin_frequency = params.get("SinFrequency")
        initial_effort = params.get("InitialEffort") / 100.0
        return _sinusoid_profile(rest_duration, ramp_duration, hold_duration, sin_frequency, initial_effort, target_effort, n_reps, samples, fs)
    elif profile_type == "Ballistic":
        ramp_duration = params.get("RampDuration")
        return _ballistic_profile(rest_duration, ramp_duration, target_effort, n_reps, samples, fs)
    else:
        # Default to constant effort
        return np.tile(np.full(samples, target_effort), n_reps)


def _create_angle_profile(params: Dict, samples: int, fs: float, movement_dof: str) -> np.ndarray:
    """Create angle profile based on parameters."""
    profile_type = params.get("AngleProfile", "Constant")

    # Read parameters from config
    rest_duration = params.get("RestDuration")
    initial_angle = params.get("InitialAngle")
    target_angle = params.get("TargetAngle")
    n_reps = params.get("NRepetitions", 1)
    
    if profile_type == "Constant":
        return np.tile(np.full(samples, target_angle), n_reps)
    elif profile_type == "Triangular":
        ramp_duration = params.get("RampDuration")
        return _triangular_angle_profile(rest_duration, ramp_duration, initial_angle, target_angle, n_reps, samples, fs, movement_dof)
    elif profile_type == "Sinusoid":
        sin_frequency = params.get("SinFrequency")
        return _sinusoid_angle_profile(sin_frequency, initial_angle, target_angle, n_reps, samples, fs, movement_dof)
    else:
        raise ValueError(f"Unsupported angle profile type: '{profile_type}'. Only 'Constant', 'Triangular', and 'Sinusoid' are supported.")


def _trapezoid_profile(rest_duration: float, ramp_duration: float, hold_duration: float, target_effort: float, n_reps: int, samples: int, fs: float) -> np.ndarray:
    """Create trapezoidal effort profile."""
    rest_samples = int(fs * rest_duration)
    ramp_samples = int(fs * ramp_duration)
    hold_samples = int(fs * hold_duration)
    
    profile = np.concatenate([
        np.zeros(rest_samples),
        np.linspace(0, target_effort, ramp_samples),
        np.full(hold_samples, target_effort),
        np.linspace(target_effort, 0, ramp_samples),
        np.zeros(rest_samples)
    ])
    
    profile = np.tile(profile, n_reps)
    return _adjust_length(profile, samples, fs)


def _triangular_profile(rest_duration: float, ramp_duration: float, n_reps: int, target_effort: float, samples: int, fs: float) -> np.ndarray:
    """Create triangular effort profile."""
    rest_samples = int(fs * rest_duration)
    ramp_samples = int(fs * ramp_duration)
    
    one_cycle = np.concatenate([
        np.zeros(rest_samples),
        np.linspace(0, target_effort, ramp_samples),
        np.linspace(target_effort, 0, ramp_samples),
        np.zeros(rest_samples)
    ])
    
    profile = np.tile(one_cycle, n_reps)
    return _adjust_length(profile, samples, fs)


def _sinusoid_profile(rest_duration: float, ramp_duration: float, hold_duration: float, sin_frequency: float, initial_effort: float, target_effort: float, n_reps: int, samples: int, fs: float) -> np.ndarray:
    """Create sinusoidal effort profile."""
    t = np.arange(samples) / fs
    rest_samples = int(fs * rest_duration)
    ramp_samples = int(fs * ramp_duration)
    hold_samples = int(fs * hold_duration)

    offset = (target_effort + initial_effort)/2
    amplitude = (target_effort - initial_effort)/2

    profile = np.concatenate([
        np.zeros(rest_samples),
        np.linspace(initial_effort, target_effort, ramp_samples),
        offset + amplitude * np.sin(2 * np.pi * sin_frequency * t[ramp_samples:ramp_samples+hold_samples] - np.pi/2),
        np.linspace(target_effort, initial_effort, ramp_samples),
        np.zeros(rest_samples),
    ])
    
    profile = np.clip(np.tile(profile, n_reps), 0, 1)
    return _adjust_length(profile, samples, fs)


def _ballistic_profile(rest_duration: float, ramp_duration: float, target_effort: float, n_reps: int, samples: int, fs: float) -> np.ndarray:
    """Create ballistic effort profile."""
    rest_samples = int(fs * rest_duration)
    ramp_samples = int(fs * ramp_duration)
    
    one_cycle = np.concatenate([
        np.zeros(rest_samples),
        np.linspace(0, target_effort, ramp_samples),
        np.zeros(rest_samples)
    ])
    
    profile = np.tile(one_cycle, n_reps)
    return _adjust_length(profile, samples, fs)


def _triangular_angle_profile(rest_duration: float, ramp_duration: float, initial_angle: float, target_angle: float, n_reps: int, samples: int, fs: float, movement_dof: str) -> np.ndarray:
    """Create triangular angle profile."""
    rest_samples = int(fs * rest_duration)
    ramp_samples = int(fs * ramp_duration)
    
    one_cycle = np.concatenate([
        np.zeros(rest_samples),
        np.linspace(initial_angle, target_angle, ramp_samples),
        np.linspace(target_angle, initial_angle, ramp_samples),
        np.zeros(rest_samples)
    ])
    
    profile = np.tile(one_cycle, n_reps)
    profile = _adjust_length(profile, samples, fs)
    min_angle, max_angle = _get_angle_range(movement_dof)
    return np.clip(profile, min_angle, max_angle)


def _sinusoid_angle_profile(sin_frequency: float, initial_angle: float, target_angle: float, n_reps: int, samples: int, fs: float, movement_dof: str) -> np.ndarray:
    """Create sinusoidal angle profile."""
    t = np.arange(samples) / fs
    offset = (target_angle + initial_angle)/2
    amplitude = (target_angle - initial_angle)/2
    profile = offset + amplitude * np.sin(2 * np.pi * sin_frequency * t - np.pi/2)
    
    profile = np.tile(profile, n_reps)
    profile = _adjust_length(profile, samples, fs)
    min_angle, max_angle = _get_angle_range(movement_dof)
    return np.clip(profile, min_angle, max_angle)


def _get_angle_range(movement_dof: str) -> Tuple[float, float]:
    """Get angle range for movement degree of freedom."""
    if movement_dof == "Flexion-Extension":
        return -65, 65
    elif movement_dof == "Radial-Ulnar-Deviation":
        return -10, 25
    else:
        raise ValueError(f"Unsupported movement DOF: '{movement_dof}'. Only 'Flexion-Extension' and 'Radial-Ulnar-Deviation' are supported.")


def _adjust_length(profile: np.ndarray, target_length: int, fs: float) -> np.ndarray:
    """Adjust profile length to target length."""
    if len(profile) > target_length:
        warnings.warn(f"Profile duration {len(profile)/fs} is greater than specified MovementDuration {target_length/fs}. Truncating profile.")
        return profile[:target_length]
    elif len(profile) < target_length:
        warnings.warn(f"Profile duration {len(profile)/fs} is less than specified MovementDuration {target_length/fs}. Padding profile.")
        return np.pad(profile, (0, target_length - len(profile)), "constant")
    else:
        return profile 