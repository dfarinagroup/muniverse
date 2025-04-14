import json
import os
import numpy as np
from scipy.stats.qmc import LatinHypercube

MUSCLE_LABELS = ["ECRB", "ECRL", "PL", "FCU", "ECU", "EDCI", "FDSI", "FCU_u"]
MOVEMENT_TYPES = ["Isometric", "Dynamic"]
MOVEMENT_TYPE_PROBS = [0.7, 0.3]
MOVEMENT_DOFS = ["Flexion-extension", "Radial-ulnar-deviation"]
MOVEMENT_DOF_PROBS = [0.5, 0.5]
MOVEMENT_PROFILES = ["Trapezoid", "Triangular", "Sinusoid"]
MOVEMENT_PROFILE_PROBS = [0.5, 0.25, 0.25]
NROW_CHOICES = [5, 10, 32]

PARAM_RANGES = {
    "SubjectSeed": (0, 4),
    "FibreDensity": (150, 250),
    "TargetMuscle": (0, 7),
    "MovementType": (0, 1),
    "MovementDOF": (0, 1),
    "MovementDuration": (20, 120),
    "EffortLevel": (5, 80),
    "MovementProfile": (0, 2),
    "InitialAngle": (0, 1),
    "FinalAngle": (0, 1),
    "PreparatoryPeriod": (1, 3),
    "NRepetitions": (1, 10),
    "RampDuration": (2, 5), 
    "HoldDuration": (15, 20),
    "SinFrequency": (0.2, 1),
    "SinAmplitude": (5, 15),
    "RampDuration": (10, 20),
    "NRows": (5, 32),
    "NoiseSeed": (1, 1000),
    "NoiseLeveldb": (10, 30),
}


def scale_sample(sample, param_ranges, param_probs=None):
    scaled = {}
    for i, (key, (low, high)) in enumerate(param_ranges.items()):
        val = sample[i] * (high - low) + low
        scaled[key] = val
    return scaled


def update_template(template, params):
    # Update SubjectConfiguration
    template["SubjectConfiguration"]["SubjectSeed"] = int(round(params["SubjectSeed"]))
    template["SubjectConfiguration"]["FibreDensity"] = float(params["FibreDensity"])

    # Update MovementConfiguration
    profile = MOVEMENT_PROFILES[int(params["MovementProfile"])]
    movement_config = template["MovementConfiguration"]
    movement_config["TargetMuscle"] = MUSCLE_LABELS[int(params["TargetMuscle"])]
    movement_config["MovementType"] = MOVEMENT_TYPES[int(params["MovementType"])]
    movement_config["MovementDOF"] = MOVEMENT_DOFS[int(params["MovementDOF"])]
    movement_config["MovementDuration"] = params["MovementDuration"]
    movement_config["EffortLevel"] = params["EffortLevel"]
    movement_config["MovementProfile"] = params["MovementProfile"]
    movement_config["MovementProfileParameters"] = {
        "InitialAngle": float(params["InitialAngle"]),
        "FinalAngle": float(params["FinalAngle"]),
        "PreparatoryPeriod": int(round(params["PreparatoryPeriod"])),
        "NRepetitions": int(round(params["NRepetitions"]))
    }

    if profile == "Trapezoid":
        movement_config["MovementProfileParameters"].update({
            "RampDuration": int(round(params["TrapezoidRampDuration"])),
            "HoldDuration": int(round(params["TrapezoidHoldDuration"]))
        })
    elif profile == "Sinusoid":
        movement_config["MovementProfileParameters"].update({
            "SinFrequency": float(params["SinFrequency"]),
            "SinAmplitude": int(round(params["SinAmplitude"]))
        })
    elif profile == "Triangular":
        movement_config["MovementProfileParameters"].update({
            "RampDuration": int(round(params["TriangularRampDuration"]))
        })

    # Update RecordingConfiguration
    template["RecordingConfiguration"]["NoiseSeed"] = int(round(params["NoiseSeed"]))
    template["RecordingConfiguration"]["NoiseLeveldb"] = int(round(params["NoiseLeveldb"]))

    # Update ElectrodeConfiguration
    template["ElectrodeConfiguration"]["NRows"] = NROW_CHOICES[int(params["NRows"])]

    return template

def generate_samples_from_template(template_path, output_dir="configs", n_samples=10):

    with open(template_path, "r") as f:
        base_template = json.load(f)

    sampler = LatinHypercube(d=len(PARAM_RANGES), seed=42)
    sample_matrix = sampler.random(n=n_samples)

    param_keys = list(PARAM_RANGES.keys())
    for i, sample in enumerate(sample_matrix):
        scaled = scale_sample(sample, PARAM_RANGES)
        config = update_template(base_template.copy(), scaled)
        with open(os.path.join(output_dir, f"config_{i:03d}.json"), "w") as f:
            json.dump(config, f, indent=2)

# Example usage
generate_samples_from_template("neuromotion_config_template.json", n_samples=10)
