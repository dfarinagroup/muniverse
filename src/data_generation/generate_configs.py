import json
import os
import numpy as np
from scipy.stats.qmc import LatinHypercube

MUSCLE_LABELS = ["ECRB", "ECRL", "ECU", "EDCI", "PL", "FCU", "FCU_u", "FDSI"]
MOVEMENT_DOFS = ["Flexion-Extension", "Radial-Ulnar-deviation"]
MOVEMENT_ANGLE_RANGES = [(-65, 65), (-10, 25)]
MOVEMENT_DOF_PROBS = [0.5, 0.5]
MOVEMENT_PROFILES = ["Trapezoid_Isometric", "Triangular_Isometric", "Ballistic_Isometric", "Sinusoid_Isometric", "Triangular_Dynamic", "Sinusoid_Dynamic"]
MOVEMENT_PROFILE_PROBS = [0.7*0.5, 0.7*0.125, 0.7*0.125, 0.7*0.25, 0.3*0.5, 0.3*0.5]
NROW_CHOICES = [5, 10, 32]

PARAM_RANGES = {
    "SubjectSeed": (0, 4),          # index, unitless (int)
    "FibreDensity": (150, 250),     # fibres/motor unit (int)
    "TargetMuscle": (0, 7),         # index, unitless (int)
    "MovementType": (0, 1),         # index, unitless (int)
    "MovementDOF": (0, 1),          # index, unitless (int)    
    "MovementProfile": (0, 2),      # index, unitless (int)
    "NRows": (5, 32),               # unitless (int)
    "NoiseSeed": (1, 1000),         # unitless (int)
    "NoiseLeveldb": (10, 30),       # dB (int)
}

PARAM_RANGES_TRAPEZOID_ISO = {
    "EffortLevel": (5, 80),         # % MVC (int)
    "RestDuration": (1, 3),         # s (int)
    "RampDuration": (5, 10),        # s (int)
    "HoldDuration": (15, 30),       # s (int)
}

PARAM_RANGES_SINUSOID_ISO = {
    "EffortLevel": (15, 80),        # % MVC (int)
    "RestDuration": (1, 3),         # s (int)
    "HoldDuration": (15, 30),       # s (int)
    "RampDuration": (5, 10),        # s (int)
    "SinFrequency": (0.025, 0.5),   # Hz (float) - range equivalent to 40s to 2s period
    "SinAmplitude": (5, 15),        # % MVC (int)
}

PARAM_RANGES_TRIANGULAR_ISO = {
    "EffortLevel": (5, 80),         # % MVC (int)
    "RestDuration": (1, 3),         # s (int)
    "RampDuration": (1, 20),        # s (int)
}

PARAM_RANGES_BALLISTIC_ISO = {
    "EffortLevel": (40, 100),       # % MVC (int)
    "NRepetitions": (1, 30),        # unitless, (int)
}

PARAM_RANGES_SINUSOID_DYN = {
    "EffortLevel": (5, 80),                # % MVC (int)
    "TargetAnglePercentage": (0.5, 1),     # % maximum angle per DOF (float)
    "TargetAngleDirection": (0, 1),        # index, unitless (int)
    "SinFrequency": (0.025, 0.5),          # Hz (float) - range equivalent to 40s to 2s period
    "SinAmplitude": (0.1, 0.5),            # % max angle for DOF (float)
}

PARAM_RANGES_TRIANGULAR_DYN = {
    "EffortLevel": (5, 80),                # % MVC (int)
    "TargetAnglePercentage": (0.3, 1),     # % maximum angle per DOF (float)
    "TargetAngleDirection": (0, 1),        # index, unitless (int)
    "RampDuration": (1, 20),               # s (float)
    "NRepetitions": (1, 10),               # unitless, (int)
}

PROFILE_PARAMS = {
    "Trapezoid_Isometric": PARAM_RANGES_TRAPEZOID_ISO,
    "Sinusoid_Isometric": PARAM_RANGES_SINUSOID_ISO,
    "Triangular_Isometric": PARAM_RANGES_TRIANGULAR_ISO,
    "Ballistic_Isometric": PARAM_RANGES_BALLISTIC_ISO,
    "Sinusoid_Dynamic": PARAM_RANGES_SINUSOID_DYN,
    "Triangular_Dynamic": PARAM_RANGES_TRIANGULAR_DYN,
}

MOVEMENT_MUSCLES = {
    "Flexion": ["FCU", "FCU_u", "FDSI", "PL"],
    "Extension": ["ECRB", "ECRL", "EDCI", "ECU"],
    "Radial": ["ECRB", "ECRL"],
    "Ulnar": ["FCU", "FCU_u", "ECU"],
}

def scale_sample(sample, param_ranges, param_probs=None):
    scaled = {}
    for i, (key, (low, high)) in enumerate(param_ranges.items()):
        val = sample[i] * (high - low) + low
        scaled[key] = val
    return scaled

def get_target_angle_props(params):
    target_angle_range = MOVEMENT_ANGLE_RANGES[int(params["MovementDOF"])]
    target_direction = int(round(params["TargetAngleDirection"]))
    mov_label = MOVEMENT_DOFS[int(params["MovementDOF"])].split("-")[target_direction]
    target_angle = target_angle_range[target_direction] * float(params["TargetAnglePercentage"])
    angle_sin_amplitude = None
    if "SinAmplitude" in params:
        angle_sin_amplitude = target_angle_range[target_direction] * float(params["SinAmplitude"])
    return mov_label, target_angle, angle_sin_amplitude

def update_template(template, params):
    # Update SubjectConfiguration
    template["SubjectConfiguration"]["SubjectSeed"] = int(round(params["SubjectSeed"]))
    template["SubjectConfiguration"]["FibreDensity"] = float(params["FibreDensity"])

    # Update MovementConfiguration
    movement_profile = MOVEMENT_PROFILES[int(params["MovementProfile"])]
    movement_type = "Isometric" if movement_profile.endswith("Isometric") else "Dynamic"
    movement_config = template["MovementConfiguration"]
    movement_config["TargetMuscle"] = MUSCLE_LABELS[int(params["TargetMuscle"])]
    movement_config["MovementType"] = movement_type
    movement_config["MovementDOF"] = MOVEMENT_DOFS[int(params["MovementDOF"])]
    movement_config["MovementProfile"] = movement_profile
    movement_config["MovementDuration"] = int(
        (round(params["RestDuration"])*2 + round(params["RampDuration"])*2 + round(params["HoldDuration"])) * round(params["NRepetitions"])
    )
    # Common properties for isometric and dynamic
    if movement_type == "Isometric":
        movement_config["MovementProfileParameters"] = {
            "EffortLevel": params["EffortLevel"],
            "RestDuration": int(round(params["RestDuration"])),
            "RampDuration": int(round(params["RampDuration"])),
        }
    elif movement_type == "Dynamic":
        mov_label, target_angle, angle_sin_amplitude = get_target_angle_props(params)
        movement_config["MovementProfileParameters"] = {
            "EffortLevel": params["EffortLevel"],
            "EffortProfile": "Constant",
            "TargetAngle": target_angle,
        }
        # Check target muscle makes sense for dynamic movement, if not, draw a random one from the list
        if params["TargetMuscle"] not in MOVEMENT_MUSCLES[mov_label]:
            movement_config["TargetMuscle"] = np.random.choice(MOVEMENT_MUSCLES[mov_label])

    # Movement specific properties
    if movement_profile == "Trapezoid_Isometric":
        movement_config["MovementProfileParameters"].update({
            "EffortProfile": "Trapezoid",
            "HoldDuration": int(round(params["HoldDuration"])),
        })
    elif movement_profile == "Sinusoid_Isometric":
        movement_config["MovementProfileParameters"].update({
            "EffortProfile": "Sinusoid",
            "HoldDuration": int(round(params["HoldDuration"])),
            "SinFrequency": float(params["SinFrequency"]),
            "SinAmplitude": int(round(params["SinAmplitude"])),
        })
    elif movement_profile == "Triangular_Isometric":
        movement_config["MovementProfileParameters"].update({
            "EffortProfile": "Triangular",
            "HoldDuration": int(0),
        })
    elif movement_profile == "Ballistic_Isometric":
        movement_config["MovementProfileParameters"].update({
            "EffortProfile": "Ballistic",
            "RampDuration": int(1),
            "HoldDuration": int(0),
            "NRepetitions": int(round(params["NRepetitions"])),
        })
    elif movement_profile == "Triangular_Dynamic":
        movement_config["MovementProfileParameters"].update({
            "AngleProfile": "Triangular",
            "RampDuration": int(round(params["RampDuration"])),
            "NRepetitions": int(round(params["NRepetitions"])),
        })
    elif movement_profile == "Sinusoid_Dynamic":
        movement_config["MovementProfileParameters"].update({
            "AngleProfile": "Sinusoid",
            "SinFrequency": float(params["SinFrequency"]),
            "SinAmplitude": angle_sin_amplitude,
        })

    # Update RecordingConfiguration
    template["RecordingConfiguration"]["NoiseSeed"] = int(round(params["NoiseSeed"]))
    template["RecordingConfiguration"]["NoiseLeveldb"] = int(round(params["NoiseLeveldb"]))

    # Update ElectrodeConfiguration
    template["ElectrodeConfiguration"]["NRows"] = NROW_CHOICES[int(params["NRows"])]

    return template

def generate_samples_from_template(template_path, output_dir="configs", n_samples=10):
    # Ensure minimum number of samples is equal to the number of movement conditions
    if n_samples <  len(MOVEMENT_PROFILES):
        n_samples = len(MOVEMENT_PROFILES)
        raise Warning(f"n_samples was set to {n_samples} to ensure all movement conditions are covered.")

    # Get number of samples per movement profile
    n_samples_per_profile = int(round(n_samples * MOVEMENT_PROFILE_PROBS))

    for profile, n_subsamples in zip(MOVEMENT_PROFILES, n_samples_per_profile):
        
        # Load template
        with open(template_path, "r") as f:
            base_template = json.load(f)

        # Sample common hyperparameters
        sampler_common = LatinHypercube(d=len(PARAM_RANGES), seed=42)
        sample_matrix_common = sampler_common.random(n=n_subsamples)

        # Sample common hyperparameters
        sampler_specific = LatinHypercube(d=len(PROFILE_PARAMS[profile]), seed=42)
        sample_matrix_specific = sampler_specific.random(n=n_subsamples)

        for i, (sample_common, sample_specific) in enumerate(zip(sample_matrix_common, sample_matrix_specific)):
            scaled_common = scale_sample(sample_common, PARAM_RANGES)
            scaled_specific = scale_sample(sample_specific, PROFILE_PARAMS[profile])
            scaled_common.update(scaled_specific)
            config = update_template(base_template.copy(), scaled_common)
            with open(os.path.join(output_dir, f"config_{i:03d}.json"), "w") as f:
                json.dump(config, f, indent=2)

# Example usage
generate_samples_from_template("neuromotion_config_template.json", n_samples=10)
