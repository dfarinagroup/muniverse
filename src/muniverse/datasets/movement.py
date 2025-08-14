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
    fs = config.get("RecordingConfiguration", {}).get("SamplingFrequency")
    params = config.get("MovementConfiguration", {}).get("MovementProfileParameters")
    duration = params.get("MovementDuration")
    
    return _create_effort_profile(params, int(duration * fs), fs)


def generate_angle_profile(config: Dict) -> np.ndarray:
    """
    Generate angle profile from config.
    
    Args:
        config: Configuration dict with MovementConfiguration and RecordingConfiguration
        
    Returns:
        Angle profile array
    """
    fs = config.get("RecordingConfiguration", {}).get("SamplingFrequency")
    params = config.get("MovementConfiguration", {}).get("MovementProfileParameters")
    duration = params.get("MovementDuration")
    movement_dof = config.get("MovementConfiguration", {}).get("MovementDOF")
    
    angle_profile = _create_angle_profile(params, int(duration * fs), fs, movement_dof)
    
    return angle_profile


def _create_effort_profile(params: Dict, samples: int, fs: float) -> np.ndarray:
    """Create effort profile based on parameters."""
    effort_level = params.get("EffortLevel", 50) / 100.0
    profile_type = params.get("EffortProfile", "Trapezoid")
    
    if profile_type == "Trapezoid":
        return _trapezoid_profile(params, samples, fs, effort_level)
    elif profile_type == "Triangular":
        return _triangular_profile(params, samples, fs, effort_level)
    elif profile_type == "Sinusoid":
        return _sinusoid_profile(params, samples, fs, effort_level)
    elif profile_type == "Ballistic":
        return _ballistic_profile(params, samples, fs, effort_level)
    else:
        # Default to constant effort
        return np.full(samples, effort_level)


def _create_angle_profile(params: Dict, samples: int, fs: float, movement_dof: str) -> np.ndarray:
    """Create angle profile based on parameters."""
    profile_type = params.get("AngleProfile", "Constant")
    target_angle = params.get("TargetAngle", 0)
    
    if profile_type == "Constant":
        return np.full(samples, target_angle)
    elif profile_type == "Triangular":
        return _triangular_profile(params, samples, fs, target_angle)
    elif profile_type == "Sinusoid":
        return _sinusoid_angle_profile(params, samples, fs, target_angle, movement_dof)
    else:
        return np.full(samples, target_angle)


def _trapezoid_profile(params: Dict, samples: int, fs: float, target: float) -> np.ndarray:
    """Create trapezoidal effort profile."""
    rest_duration = params.get("RestDuration", 1)
    ramp_duration = params.get("RampDuration", 5)
    hold_duration = params.get("HoldDuration", 10)
    
    rest_samples = int(fs * rest_duration)
    ramp_samples = int(fs * ramp_duration)
    hold_samples = int(fs * hold_duration)
    
    profile = np.concatenate([
        np.zeros(rest_samples),
        np.linspace(0, target, ramp_samples),
        np.full(hold_samples, target),
        np.linspace(target, 0, ramp_samples),
        np.zeros(rest_samples)
    ])
    
    return _adjust_length(profile, samples, fs)


def _triangular_profile(params: Dict, samples: int, fs: float, target: float) -> np.ndarray:
    """Create triangular effort profile."""
    rest_duration = params.get("RestDuration", 1)
    ramp_duration = params.get("RampDuration", 5)
    n_reps = params.get("NRepetitions", 1)
    
    rest_samples = int(fs * rest_duration)
    ramp_samples = int(fs * ramp_duration)
    
    one_cycle = np.concatenate([
        np.zeros(rest_samples),
        np.linspace(0, target, ramp_samples),
        np.linspace(target, 0, ramp_samples),
        np.zeros(rest_samples)
    ])
    
    profile = np.tile(one_cycle, n_reps)
    return _adjust_length(profile, samples, fs)


def _sinusoid_profile(params: Dict, samples: int, fs: float, target: float) -> np.ndarray:
    """Create sinusoidal effort profile."""
    sin_frequency = params.get("SinFrequency", 0.2)
    sin_amplitude = params.get("SinAmplitude", 25) / 100.0
    rest_duration = params.get("RestDuration", 0)
    
    t = np.arange(samples) / fs
    profile = target + sin_amplitude * np.sin(2 * np.pi * sin_frequency * t)
    
    if rest_duration > 0:
        rest_samples = int(fs * rest_duration)
        profile[:rest_samples] = 0
    
    return np.clip(profile, 0, 1)


def _ballistic_profile(params: Dict, samples: int, fs: float, target: float) -> np.ndarray:
    """Create ballistic effort profile."""
    rest_duration = params.get("RestDuration", 1.0)
    n_reps = params.get("NRepetitions", 1)
    ramp_duration = params.get("RampDuration", 0.05)
    
    rest_samples = int(fs * rest_duration)
    ramp_samples = int(fs * ramp_duration)
    
    one_cycle = np.concatenate([
        np.zeros(rest_samples),
        np.linspace(0, target, ramp_samples),
        np.zeros(rest_samples)
    ])
    
    profile = np.tile(one_cycle, n_reps)
    return _adjust_length(profile, samples, fs)


def _sinusoid_angle_profile(params: Dict, samples: int, fs: float, target: float, movement_dof: str) -> np.ndarray:
    """Create sinusoidal angle profile."""
    sin_amplitude = params.get("SinAmplitude", 0.3)
    sin_frequency = params.get("SinFrequency", 0.2)
    
    t = np.arange(samples) / fs
    profile = target + sin_amplitude * np.sin(2 * np.pi * sin_frequency * t)
    
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