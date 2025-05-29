import json
import os
import numpy as np
from scipy.stats.qmc import LatinHypercube

def get_deterministic_mu_count(seed, min_mus=300, max_mus=380):
    """
    Get a deterministic number of motor units based on seed.
    
    Args:
        seed (int): Random seed
        min_mus (int): Minimum number of motor units
        max_mus (int): Maximum number of motor units
        
    Returns:
        int: Number of motor units
    """
    # Create dedicated RNG just for this operation
    rng = np.random.RandomState(seed)
    return rng.randint(min_mus, max_mus + 1)

def generate_configs_hybrid_tibialis(template_path, output_dir="configs", n_subjects=5, n_contractions_per_subject=20):
    """
    Generates configuration files for hybrid tibialis setup.
    
    Args:
        template_path (str): Path to the template JSON file
        output_dir (str): Directory to save the generated configs
        n_subjects (int): Number of subjects to generate (default: 5)
        n_contractions_per_subject (int): Number of contractions per subject (default: 20)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define movement profiles and their probabilities
    MOVEMENT_PROFILES = ["Trapezoid_Isometric", "Triangular_Isometric"]
    MOVEMENT_PROFILE_PROBS = [0.5, 0.5]  # Equal probability for each profile
    
    # Common parameters
    COMMON_PARAM_RANGES = {
        "NoiseSeed": (1, 1000),         # unitless (int)
        "NoiseLeveldb": (10, 30),       # dB (int)
    }
    
    # Profile-specific parameters - UPDATED as requested
    PARAM_RANGES_TRAPEZOID_ISO = {
        "EffortLevel": (5, 80),         # % MVC (int)
        "RestDuration": (1, 3),         # s (int)
        "RampDuration": (1, 8),         # s (int)
        "HoldDuration": (15, 30),       # s (int) 
    }
    
    PARAM_RANGES_TRIANGULAR_ISO = {
        "EffortLevel": (5, 80),         # % MVC (int)
        "RestDuration": (1, 3),         # s (int)
        "RampDuration": (10, 20),       # s (int) 
    }
    
    PROFILE_PARAMS = {
        "Trapezoid_Isometric": PARAM_RANGES_TRAPEZOID_ISO,
        "Triangular_Isometric": PARAM_RANGES_TRIANGULAR_ISO,
    }
    
    # Calculate number of contractions per profile
    total_contractions = n_subjects * n_contractions_per_subject
    n_contractions_per_profile = [round(total_contractions * prob) for prob in MOVEMENT_PROFILE_PROBS]
    
    # Adjust if rounding error causes mismatch
    if sum(n_contractions_per_profile) != total_contractions:
        diff = total_contractions - sum(n_contractions_per_profile)
        n_contractions_per_profile[0] += diff
    
    # Generate configs for each subject
    config_id = 0
    for subject_id in range(1, n_subjects + 1):
        # Get deterministic number of motor units for this subject
        num_motor_units = get_deterministic_mu_count(subject_id)
        print(f"Subject {subject_id:02d}: Using {num_motor_units} motor units")
        
        # Distribute profiles evenly across subjects
        contractions_per_profile_per_subject = [n // n_subjects for n in n_contractions_per_profile]
        
        # Adjust if division causes rounding issues
        extra_contractions = [n % n_subjects for n in n_contractions_per_profile]
        for i, extra in enumerate(extra_contractions):
            if subject_id <= extra:
                contractions_per_profile_per_subject[i] += 1
        
        # Generate contractions for each profile
        for profile_idx, profile in enumerate(MOVEMENT_PROFILES):
            n_contractions = contractions_per_profile_per_subject[profile_idx]
            
            if n_contractions <= 0:
                continue
                
            # Sample common parameters
            sampler_common = LatinHypercube(d=len(COMMON_PARAM_RANGES), seed=42 + subject_id)
            sample_matrix_common = sampler_common.random(n=n_contractions)
            
            # Sample profile-specific parameters
            profile_params = PROFILE_PARAMS[profile]
            sampler_specific = LatinHypercube(d=len(profile_params), seed=42 + subject_id + 100)
            sample_matrix_specific = sampler_specific.random(n=n_contractions)
            
            for idx in range(n_contractions):
                # Load fresh template for each config
                with open(template_path, "r") as f:
                    config = json.load(f)
                
                # Scale and update common parameters
                common_params = scale_sample(sample_matrix_common[idx], COMMON_PARAM_RANGES)
                
                # Scale and update profile-specific parameters
                specific_params = scale_sample(sample_matrix_specific[idx], profile_params)
                
                # Update subject configuration
                config["SubjectConfiguration"]["SubjectSeed"] = subject_id
                
                # Update the motor unit count in the MuscleMotorUnitCounts array
                config["SubjectConfiguration"]["MuscleMotorUnitCounts"] = [num_motor_units]
                
                # Update movement configuration
                config["MovementConfiguration"]["TargetMuscle"] = "Tibialis Anterior"
                config["MovementConfiguration"]["MovementType"] = "Isometric"
                config["MovementConfiguration"]["MovementDOF"] = "Ankle Dorsiflexion"
                
                # Update movement profile parameters based on the selected profile
                movement_params = config["MovementConfiguration"]["MovementProfileParameters"]
                
                # Set effort level and rest duration for all profiles
                movement_params["EffortLevel"] = specific_params["EffortLevel"]
                movement_params["RestDuration"] = specific_params["RestDuration"]
                
                # Profile-specific settings
                if profile == "Trapezoid_Isometric":
                    movement_params["EffortProfile"] = "Trapezoid"
                    movement_params["RampDuration"] = specific_params["RampDuration"]
                    movement_params["HoldDuration"] = specific_params["HoldDuration"]
                    
                    # Calculate movement duration for trapezoid
                    movement_params["MovementDuration"] = (
                        2 * movement_params["RestDuration"] + 
                        2 * movement_params["RampDuration"] + 
                        movement_params["HoldDuration"]
                    )
                    
                elif profile == "Triangular_Isometric":
                    movement_params["EffortProfile"] = "Triangular"
                    movement_params["RampDuration"] = specific_params["RampDuration"]
                    movement_params["NRepetitions"] = 1  # Fixed at 1 as requested
                    movement_params["HoldDuration"] = 0  # Set to 0 for triangular
                    
                    # Calculate movement duration for triangular with fixed NRepetitions=1
                    movement_params["MovementDuration"] = (
                        2 * movement_params["RestDuration"] + 
                        2 * movement_params["RampDuration"]
                    )
                
                # Update recording configuration
                config["RecordingConfiguration"]["NoiseSeed"] = common_params["NoiseSeed"]
                config["RecordingConfiguration"]["NoiseLeveldb"] = common_params["NoiseLeveldb"]
                
                # Set MUAPs file path for the hybrid approach
                config["PathToMUAPFile"] = "./ckp/tibialis_curated_muaps.npz"
                
                # Format output filename
                n_digits = int(np.log10(total_contractions)) + 1
                filename = f"tibialis_subject_{subject_id:02d}_contraction_{config_id:0{n_digits}d}.json"
                
                # Save configuration
                with open(os.path.join(output_dir, filename), "w") as f:
                    json.dump(config, f, indent=2)
                
                config_id += 1
    
    print(f"Generated {config_id} configuration files in {output_dir}")


def scale_sample(sample, param_ranges):
    """Scale a normalized LHS sample to the specified parameter ranges."""
    scaled = {}
    for i, (key, (low, high)) in enumerate(param_ranges.items()):
        val = sample[i] * (high - low) + low
        if isinstance(low, int) and isinstance(high, int):
            scaled[key] = int(val)
        else:
            scaled[key] = round(val, 2) if isinstance(val, float) else val
    return scaled


if __name__ == "__main__":
    template_path = "configs/hybrid_tibialis.json"
    generate_configs_hybrid_tibialis(template_path, "configs/generated")