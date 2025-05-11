#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import json
from edfio import Edf, EdfSignal
from muniverse.data_preparation.data2bids import bids_dataset, bids_neuromotion_recording

NEUROMOTION_OUTPUTS_PATH = '/rds/general/user/pm1222/ephemeral/muniverse/datasets/outputs/neuromotion-test'
SIDECAR_PATH = os.path.abspath("./bids_metadata/neuromotion.json")
ROOT = '/rds/general/user/pm1222/ephemeral/muniverse/datasets/bids/'
DATASETNAME = "neuromotion-test"
MAX_RUNS = 0

def find_file_by_suffix(data_path, suffix):
    """Find a file ending with the given suffix in the data path"""
    for file in os.listdir(data_path):
        if file.endswith(suffix):
            return os.path.join(data_path, file)
    raise FileNotFoundError(f"No file ending with '{suffix}' found in {data_path}")

def setup_spikes_data(data_path):
    """Load and format spike times"""
    spikes_file = find_file_by_suffix(data_path, '_spikes.npz')
    spikes = np.load(spikes_file, allow_pickle=True)['spikes']
    
    # Convert to long format DataFrame
    rows = []
    for unit_id, spike_times in enumerate(spikes):
        for t in spike_times:
            rows.append({'source_id': unit_id, 'spike_time': t})
            
    return pd.DataFrame(rows).sort_values('spike_time')

def setup_motor_units_data(data_path):
    """Load and format motor unit properties"""
    recruitment_file = find_file_by_suffix(data_path, '_recruitment_thresholds.npy')
    properties_file = find_file_by_suffix(data_path, '_unit_properties.npy')
    
    recruitment_thresholds = np.load(recruitment_file, allow_pickle=True)
    unit_properties = np.load(properties_file, allow_pickle=True)
    
    return pd.DataFrame({
        'unit_id': np.arange(len(recruitment_thresholds)),
        'recruitment_threshold': recruitment_thresholds,
        'depth': unit_properties[:,0],
        'innervation_zone': unit_properties[:,1],
        'fibre_density': unit_properties[:,2],
        'fibre_length': unit_properties[:,3],
        'conduction_velocity': unit_properties[:,4],
        'angle': unit_properties[:,5]
    })

def setup_internals_data(data_path, fsamp):
    """Load and format internal variables"""
    effort_file = find_file_by_suffix(data_path, '_effort_profile.npz')
    angle_file = find_file_by_suffix(data_path, '_angle_profile.npz')
    
    effort_profile = np.load(effort_file, allow_pickle=True)['effort_profile']
    angle_profile = np.load(angle_file, allow_pickle=True)['angle_profile']
    
    # Create Edf object with both signals
    edf = Edf([EdfSignal(effort_profile, sampling_frequency=fsamp)])
    edf.append_signals(EdfSignal(angle_profile, sampling_frequency=fsamp))
    
    return edf

def setup_internals_metadata():
    """Generate internal variables metadata"""
    return {
        'name': ['effort_profile', 'angle_profile'],
        'type': ['force', 'kinematics'],
        'units': ['AU', 'AU'],
        'description': ['simulated effort (excitation) profile', 'simulated angle profile']
    }

def setup_electrode_metadata(config):
    """Generate electrode metadata based on RecordingConfiguration"""
    elec_config = config['RecordingConfiguration']['ElectrodeConfiguration']
    
    names = []
    x_pos = []
    y_pos = []
    coordinate_system = []
    diameter = []
    
    for i in range(elec_config['NElectrodes']):
        row = i // elec_config['NCols']
        col = i % elec_config['NCols']
        
        names.append(f"E{i+1}")
        x_pos.append(col * elec_config['InterElectrodeDistance'])
        y_pos.append(row * elec_config['InterElectrodeDistance'])
        coordinate_system.append("Grid1")
        diameter.append("2mm")
    return {
        'name': names,
        'x': x_pos,
        'y': y_pos,
        'coordinate_system': coordinate_system,
        'diameter': diameter
    }

def setup_channel_metadata(config):
    """Generate channel metadata"""
    elec_config = config['RecordingConfiguration']['ElectrodeConfiguration']
    
    names = []
    types = []
    units = []
    signal_electrode = []
    reference_electrode = []
    
    for i in range(elec_config['NElectrodes']):
        names.append(f"Ch{i+1}")
        types.append("EMG")
        units.append("mV")
        signal_electrode.append(f"E{i+1}")
        reference_electrode.append("n/a")
        
    return {
        'name': names,
        'type': types,
        'unit': units,
        'signal_electrode': signal_electrode, 
        'reference_electrode': reference_electrode
    }

def generate_task_name(config):
    """Generate a BIDS-compatible task name from config"""
    movement = config['MovementConfiguration']
    profile = movement['MovementProfileParameters']
    muscle = movement['TargetMuscle']
    movement_type = movement['MovementType']
    movement_dof = movement['MovementDOF']
    effort_level = profile['EffortLevel']
    
    # Base task name with muscle, movement type, and DOF
    task_name = f"{muscle.upper()}{movement_type.lower()}{movement_dof.split('-')[0].lower()}"
    
    # Add movement-specific parameters
    if movement_type == 'Isometric':
        effort_profile = profile['EffortProfile']
        task_name += f"{effort_profile.lower()}"
    else:  # Dynamic
        angle_profile = profile['AngleProfile']
        task_name += f"{angle_profile.lower()}"
    
    # Add effort level and subject ID
    task_name += f"{effort_level}percentmvc"

    # Remove any non-alphanumeric characters
    task_name = ''.join(char for char in task_name if char.isalnum())
    
    return task_name

def get_subject_name(config):
    """Generate subject name from config"""
    subject_id = config['SubjectConfiguration']['SubjectSeed']
    return f"sub-sim{subject_id}"

def generate_session_id(ncols):
    """Generate session ID based on number of columns in electrode array"""
    if ncols == 32:
        return 1
    elif ncols == 10:
        return 2
    elif ncols == 5:
        return 3
    else:
        raise ValueError(f"Unsupported number of columns: {ncols}")

def generate_run_id(dataset, subject_name, task_name, session_id):
    """Generate a unique run ID for the recording"""
    run_id = 1
    while run_id < 5:
        emg_file_path = os.path.join(
            dataset.root,
            subject_name,
            'emg',
            f'ses-{session_id}',
            f'{subject_name}_ses-{session_id}_task-{task_name}_run-{run_id:02d}_emg.edf'
        )
        if not os.path.exists(emg_file_path):
            break
        run_id += 1
    return run_id

def load_sidecar_files(sidecar_path):
    """Load BIDS sidecar files from neuromotion.json"""
    with open(sidecar_path, 'r') as f:
        sidecars = json.load(f)
    return sidecars

def write_neuromotion_to_bids(data_path, sidecar_path, root='./', datasetname='simulated-BIDS-dataset'):
    """
    Convert neuromotion simulation data to BIDS format
    
    Parameters:
    -----------
    data_path : str
        Path to the directory containing neuromotion simulation outputs
    root : str
        Root directory for the BIDS dataset
    datasetname : str
        Name of the BIDS dataset
    """
    # Find and read the run log file
    run_logs = [f for f in os.listdir(data_path) if f.endswith('_log.json')]
    if not run_logs:
        raise ValueError(f"No run log file found in {data_path}")
    if len(run_logs) > 1:
        raise ValueError(f"Multiple run log files found in {data_path}: {run_logs}")
        
    with open(os.path.join(data_path, run_logs[0]), 'r') as f:
        simulation_config = json.load(f)['InputData']['Configuration']
    
    # Load sidecar files
    sidecars = load_sidecar_files(sidecar_path)
    
    # Create dataset
    dataset = bids_dataset(datasetname=datasetname, root=root)
    dataset.set_metadata('dataset_sidecar', sidecars['DatasetDescription'])
    dataset.write()
    
    # Set up subject metadata
    subject_name = get_subject_name(simulation_config)
    subjects_data = {'name': [subject_name]}
    dataset.set_metadata('subjects_data', subjects_data)
    dataset.write()
    
    # Create recording
    task_name = generate_task_name(simulation_config)
    fsamp = simulation_config['RecordingConfiguration']['SamplingFrequency']
    ncols = simulation_config['RecordingConfiguration']['ElectrodeConfiguration']['NCols']
    
    # Generate session and run IDs
    session_id = generate_session_id(ncols)
    run_id = generate_run_id(dataset, subject_name, task_name, session_id)
    
    # Create recording with new path structure
    recording = bids_neuromotion_recording(
        subject=simulation_config['SubjectConfiguration']['SubjectSeed'],
        task=task_name,
        datatype='emg',
        data_obj=dataset,
        session=session_id,
        run=run_id
    )
    
    # Set up metadata
    recording.set_metadata('electrodes', setup_electrode_metadata(simulation_config))
    recording.set_metadata('channels', setup_channel_metadata(simulation_config))
    recording.set_metadata('coord_sidecar', sidecars['CoordSystemSidecar'])
    
    # Set up EMG sidecar with simulation-specific modifications
    emg_sidecar = sidecars['EMGSidecar'].copy()
    emg_sidecar['SamplingFrequency'] = fsamp
    emg_sidecar['TaskName'] = f"Muscle: {simulation_config['MovementConfiguration']['TargetMuscle'].upper()}, " \
                            f"Task: {simulation_config['MovementConfiguration']['MovementType'].lower()}, " \
                            f"DoF: {simulation_config['MovementConfiguration']['MovementDOF']}, " \
                            f"Profile: {simulation_config['MovementConfiguration']['MovementProfileParameters']['EffortProfile'].lower()}, " \
                            f"EffortLevel: {simulation_config['MovementConfiguration']['MovementProfileParameters']['EffortLevel']} percent mvc"
    recording.set_metadata('emg_sidecar', emg_sidecar)
    
    # Set up data
    emg_file = find_file_by_suffix(data_path, '_emg.npz')
    data = np.load(emg_file, allow_pickle=True)['emg']
    recording.set_data('emg_data', data, fsamp)
    
    # Set up simulation-specific data
    recording.spikes = setup_spikes_data(data_path)
    # recording.motor_units = setup_motor_units_data(data_path)
    recording.internals = setup_internals_data(data_path, fsamp)
    recording.set_metadata('internals_sidecar', setup_internals_metadata())
    
    # Add simulation metadata
    with open(os.path.join(data_path, run_logs[0]), 'r') as f:
        simulation_metadata = json.load(f)
    recording.set_metadata('simulation_sidecar', simulation_metadata)
    
    # Write to BIDS format
    recording.write()
    
    return recording 

def process_runs(outputs_path: str, sidecar_path: str, root: str, datasetname: str, max_runs: int = None):
    """
    Process run directories and convert them to BIDS format.
    
    Args:
        outputs_path (str): Path to the outputs directory containing run directories
        sidecar_path (str): Path to the BIDS metadata sidecar file
        root (str): Root path for the dataset
        datasetname (str): Name of the dataset
        max_runs (int, optional): Maximum number of runs to process. If None, process all runs.
    """
    # Get all directories that start with 'run_'
    run_dirs = [d for d in os.listdir(outputs_path) if d.startswith('run_') and os.path.isdir(os.path.join(outputs_path, d))]
    
    # Sort the directories to process them in order
    run_dirs.sort()
    
    if max_runs:
        run_dirs = run_dirs[:max_runs]
    
    print(f"Found {len(run_dirs)} run directories to process")
    
    # Process each run directory
    for run_dir in run_dirs:
        print(f"\nProcessing {run_dir}...")
        data_path = os.path.join(outputs_path, run_dir)
        
        try:
            write_neuromotion_to_bids(data_path, sidecar_path, root, datasetname)
            print(f"Successfully converted {run_dir} to BIDS format")
        except Exception as e:
            print(f"Error processing {run_dir}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Process run directories and convert them to BIDS format')
    parser.add_argument('--outputs_path', default=NEUROMOTION_OUTPUTS_PATH,
                      help='Path to the outputs directory containing run directories')
    parser.add_argument('--sidecar_path', default=SIDECAR_PATH,
                      help='Path to the BIDS metadata sidecar file')
    parser.add_argument('--root', default=ROOT,
                      help='Root path for the dataset')
    parser.add_argument('--datasetname', default=DATASETNAME,
                      help='Name of the dataset')
    parser.add_argument('--max_runs', type=int, default=MAX_RUNS,
                      help='Maximum number of runs to process')
    
    args = parser.parse_args()
    
    print(f"Processing runs from: {args.outputs_path}")
    print(f"Using sidecar file: {args.sidecar_path}")
    print(f"Dataset name: {args.datasetname}")
    print(f"Maximum runs to process: {args.max_runs}")
    
    process_runs(
        outputs_path=args.outputs_path,
        sidecar_path=args.sidecar_path,
        root=args.root,
        datasetname=args.datasetname,
        max_runs=args.max_runs
    )
    
    print("\nProcessing complete!")

if __name__ == '__main__':
    main() 
