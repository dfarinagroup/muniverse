import json
import os
import numpy as np
from scipy.stats.qmc import LatinHypercube

MUSCLE_LABELS = ["ECRB", "ECRL", "ECU", "EDCI", "PL", "FCU", "FCU_u", "FDSI"]
MOVEMENT_DOFS = ["Flexion-Extension", "Radial-Ulnar-deviation"]
MOVEMENT_PROFILES = ["Trapezoid_Isometric", "Triangular_Isometric", "Ballistic_Isometric", "Sinusoid_Isometric", "Triangular_Dynamic", "Sinusoid_Dynamic"]
NROW_CHOICES = [5, 10, 32]
MOVEMENT_ANGLE_RANGES = [(-65, 65), (-10, 25)]

MOVEMENT_DOF_PROBS = [0.65, 0.35]
MOVEMENT_PROFILE_PROBS = [0.5*0.7, 0.125*0.7, 0.125*0.7, 0.25*0.7, 0.5*0.3, 0.5*0.3] # p(Trapezoid/Triangular/Ballistic/Sinusiod)xp(isometric/dynamic)

COMMON_PARAM_RANGES = {
    "SubjectSeed": (0, 5),          # index, unitless (int)
    "FibreDensity": (150, 250),     # fibres/motor unit (int)
    "TargetMuscle": (0, 8),         # index, unitless (int)
    "MovementDOF": (0, 2),          # index, unitless (int)    
    "NRows": (0, 3),                # unitless (int)
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
    "RestDuration": (1, 3),         # s (int)
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

def scale_sample(sample, param_ranges):
    scaled = {}
    for i, (key, (low, high)) in enumerate(param_ranges.items()):
        val = sample[i] * (high - low) + low
        scaled[key] = round(val, 2) if isinstance(val, float) else val
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
    template["SubjectConfiguration"]["SubjectSeed"] = int(params["SubjectSeed"])
    template["SubjectConfiguration"]["FibreDensity"] = float(params["FibreDensity"])

    # Update MovementConfiguration
    template["MovementConfiguration"]["TargetMuscle"] = MUSCLE_LABELS[int(params["TargetMuscle"])]
    template["MovementConfiguration"]["MovementDOF"] = MOVEMENT_DOFS[int(params["MovementDOF"])]
    
    movement_profile = params["MovementProfile"]
    movement_type = params["MovementProfile"].split("_")[1]
    template["MovementConfiguration"]["MovementType"] = movement_type

    # Update MovementProfileParameters
    template_profile = template["MovementConfiguration"]["MovementProfileParameters"]
    
    # Common properties for isometric and dynamic
    if movement_type == "Isometric":
        template_profile["EffortLevel"] = params["EffortLevel"]       
        template_profile["RestDuration"] = round(params["RestDuration"])
    
    elif movement_type == "Dynamic":
        mov_label, target_angle, angle_sin_amplitude = get_target_angle_props(params)
        template_profile["EffortLevel"] = params["EffortLevel"]
        template_profile["EffortProfile"] = "Constant"
        template_profile["TargetAngle"] = target_angle
        
        # Check target muscle makes sense for dynamic movement, if not, draw a random one from the list
        if params["TargetMuscle"] not in MOVEMENT_MUSCLES[mov_label]:
            template["MovementConfiguration"]["TargetMuscle"] = np.random.choice(MOVEMENT_MUSCLES[mov_label])

    # Movement specific properties
    if movement_profile == "Trapezoid_Isometric":
        template_profile["EffortProfile"] = "Trapezoid"
        template_profile["HoldDuration"] = round(params["HoldDuration"])
        template_profile["RampDuration"] = round(params["RampDuration"])
    
    elif movement_profile == "Sinusoid_Isometric":
        template_profile["EffortProfile"] = "Sinusoid"
        template_profile["HoldDuration"] = round(params["HoldDuration"])
        template_profile["RampDuration"] = round(params["RampDuration"])
        template_profile["SinFrequency"] = float(params["SinFrequency"])
        template_profile["SinAmplitude"] = round(params["SinAmplitude"])
    
    elif movement_profile == "Triangular_Isometric":
        template_profile["EffortProfile"] = "Triangular"
        template_profile["HoldDuration"] = 0
        template_profile["RampDuration"] = round(params["RampDuration"])
    
    elif movement_profile == "Ballistic_Isometric":
        template_profile["EffortProfile"] = "Ballistic"
        template_profile["HoldDuration"] = 0
        template_profile["RampDuration"] = 1
        template_profile["NRepetitions"] = round(params["NRepetitions"])
    
    elif movement_profile == "Triangular_Dynamic":
        template_profile["AngleProfile"] = "Triangular"
        template_profile["RampDuration"] = 1
        template_profile["NRepetitions"] = round(params["NRepetitions"])

    elif movement_profile == "Sinusoid_Dynamic":
        template_profile["AngleProfile"] = "Sinusoid"
        template_profile["SinFrequency"] = float(params["SinFrequency"])
        template_profile["SinAmplitude"] = angle_sin_amplitude

    template_profile["MovementDuration"] = (
        template_profile["RestDuration"]*2 + 
        template_profile["RampDuration"]*2 + 
        template_profile["HoldDuration"]) * template_profile["NRepetitions"]

    # Update RecordingConfiguration
    template["RecordingConfiguration"]["NoiseSeed"] = int(params["NoiseSeed"])
    template["RecordingConfiguration"]["NoiseLeveldb"] = int(params["NoiseLeveldb"])
    n_rows = int(min(2, (5/3)*params["NRows"])) # ensures that 32 Rows are chosen 60% of the time
    template["RecordingConfiguration"]["ElectrodeConfiguration"]["NRows"] = NROW_CHOICES[n_rows]
    return template

def generate_configs(template_path, output_dir="configs", n_samples=10):
    """
    Generates a specified number of configuration files for neuromotion based on a template file.
    
    """
    # Ensure minimum number of samples is equal to the number of movement conditions
    if n_samples <  len(MOVEMENT_PROFILES):
        n_samples = len(MOVEMENT_PROFILES)
        raise Warning(f"n_samples was set to {n_samples} to ensure all movement conditions are covered.")

    # Get number of samples per movement profile
    n_samples_per_profile = [round(n_samples*i) for i in MOVEMENT_PROFILE_PROBS]
    sample_id = 0

    for profile, n_subsamples in zip(MOVEMENT_PROFILES, n_samples_per_profile):
        
        # Load template
        with open(template_path, "r") as f:
            base_template = json.load(f)

        # Sample common hyperparameters
        sampler_common = LatinHypercube(d=len(COMMON_PARAM_RANGES), seed=42)
        sample_matrix_common = sampler_common.random(n=n_subsamples)

        # Sample common hyperparameters
        sampler_specific = LatinHypercube(d=len(PROFILE_PARAMS[profile]), seed=42)
        sample_matrix_specific = sampler_specific.random(n=n_subsamples)

        for sample_common, sample_specific in zip(sample_matrix_common, sample_matrix_specific):
            scaled_common = scale_sample(sample_common, COMMON_PARAM_RANGES)
            scaled_specific = scale_sample(sample_specific, PROFILE_PARAMS[profile])
            
            scaled_common.update(scaled_specific)
            scaled_common.update({"MovementProfile": profile})
            config = update_template(base_template.copy(), scaled_common)
            
            with open(os.path.join(output_dir, f"config_{sample_id:04d}.json"), "w") as f:
                json.dump(config, f, indent=2)
            sample_id += 1
