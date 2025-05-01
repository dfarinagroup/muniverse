import os
import numpy as np
import pandas as pd
import json
from edfio import *
from ..data_preparation.data2bids import bids_dataset, bids_emg_recording, bids_neuromotion_recording

def setup_spikes_data(data_path):
    """Load and format spike times"""
    spikes = np.load(os.path.join(data_path, 'spikes.npy'), allow_pickle=True)
    
    # Convert to long format DataFrame
    rows = []
    for unit_id, spike_times in enumerate(spikes):
        for t in spike_times:
            rows.append({'source_id': unit_id, 'spike_time': t})
            
    return pd.DataFrame(rows).sort_values('spike_time')

def setup_motor_units_data(data_path):
    """Load and format motor unit properties"""
    recruitment_thresholds = np.load(os.path.join(data_path, 'recruitment_thresholds.npy'), allow_pickle=True)
    unit_properties = np.load(os.path.join(data_path, 'unit_properties.npy'), allow_pickle=True)
    
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
    effort_profile = np.load(os.path.join(data_path, 'effort_profile.npy'), allow_pickle=True)
    angle_profile = np.load(os.path.join(data_path, 'angle_profile.npy'), allow_pickle=True)
    
    edf = Edf([EdfSignal(effort_profile, sampling_frequency=fsamp)])
    edf.append_signals(EdfSignal(angle_profile, sampling_frequency=fsamp))
    
    return edf

def setup_electrode_metadata(config):
    """Generate electrode metadata based on RecordingConfiguration"""
    elec_config = config['RecordingConfiguration']['ElectrodeConfiguration']
    
    names = []
    x_pos = []
    y_pos = []
    coordinate_system = []
    
    for i in range(elec_config['NElectrodes']):
        row = i // elec_config['NCols']
        col = i % elec_config['NCols']
        
        names.append(f"E{i+1}")
        x_pos.append(col * elec_config['InterElectrodeDistance'])
        y_pos.append(row * elec_config['InterElectrodeDistance'])
        coordinate_system.append("Grid1")
        
    return {
        'name': names,
        'x': x_pos,
        'y': y_pos,
        'coordinate_system': coordinate_system
    }

def setup_channel_metadata(config):
    """Generate channel metadata"""
    elec_config = config['RecordingConfiguration']['ElectrodeConfiguration']
    
    names = []
    types = []
    units = []
    
    for i in range(elec_config['NElectrodes']):
        names.append(f"E{i+1}")
        types.append("EMG")
        units.append("mV")
        
    return {
        'name': names,
        'type': types,
        'unit': units
    }

def get_task_name(config):
    """Generate a BIDS-compatible task name from config"""
    movement = config['MovementConfiguration']
    profile = movement['MovementProfileParameters']
    muscle = movement['TargetMuscle']
    movement_type = movement['MovementType']
    effort_level = profile['EffortLevel']
    if movement_type == 'Isometric':
        effort_profile = profile['EffortProfile']
    else:
        effort_profile = profile['AngleProfile']
    
    return f"{muscle.upper()}{movement_type.lower()}{effort_profile.lower()}{effort_level}percentmvc"

def get_subject_name(config):
    """Generate subject name from config"""
    subject_id = config['SubjectConfiguration']['SubjectSeed']
    return f"sub-sim{subject_id}"

def neuromotion_to_bids(data_path, root='./', datasetname='simulated-BIDS-dataset'):
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
    run_logs = [f for f in os.listdir(data_path) if f.startswith('run_log') and f.endswith('.json')]
    if not run_logs:
        raise ValueError(f"No run log file found in {data_path}")
    if len(run_logs) > 1:
        raise ValueError(f"Multiple run log files found in {data_path}: {run_logs}")
        
    with open(os.path.join(data_path, run_logs[0]), 'r') as f:
        simulation_config = json.load(f)['SimulationConfiguration']
    
    # Create dataset
    dataset = bids_dataset(datasetname=datasetname, root=root)
    
    # Set up subject metadata
    subject_name = get_subject_name(simulation_config)
    subjects_data = {'name': [subject_name]}
    dataset.set_metadata('subjects_data', subjects_data)
    dataset.write()
    
    # Create recording
    task_name = get_task_name(simulation_config)
    fsamp = simulation_config['RecordingConfiguration']['SamplingFrequency']
    
    recording = bids_neuromotion_recording(
        subject=simulation_config['SubjectConfiguration']['SubjectSeed'],
        task=task_name,
        datatype='emg',
        data_obj=dataset
    )
    
    # Set up metadata
    recording.set_metadata('electrodes', setup_electrode_metadata(simulation_config))
    recording.set_metadata('channels', setup_channel_metadata(simulation_config))
    recording.set_metadata('coord_sidecar', {'EMGCoordinateSystem': 'local', 'EMGCoordinateUnits': 'mm'})
    
    # Set up EMG sidecar
    emg_sidecar = {
        'EMGPlacementScheme': 'grid',
        'EMGReference': 'bipolar',
        'SamplingFrequency': fsamp,
        'PowerLineFrequency': 50,
        'SoftwareFilters': [],
        'TaskName': f"Muscle: {simulation_config['MovementConfiguration']['TargetMuscle'].upper()}, "
                   f"Task: {simulation_config['MovementConfiguration']['MovementType'].lower()}, "
                   f"DoF: {simulation_config['MovementConfiguration']['MovementDOF']}, "
                   f"Profile: {simulation_config['MovementConfiguration']['MovementProfileParameters']['EffortProfile'].lower()}, "
                   f"EffortLevel: {simulation_config['MovementConfiguration']['MovementProfileParameters']['EffortLevel']} percent mvc"
    }
    recording.set_metadata('emg_sidecar', emg_sidecar)
    
    # Set up data
    data = np.load(os.path.join(data_path, 'emg.npy'), allow_pickle=True)
    recording.set_data('emg_data', data, fsamp)
    
    # Set up simulation-specific data
    recording.spikes = setup_spikes_data(data_path)
    # recording.motor_units = setup_motor_units_data(data_path)
    recording.internals = setup_internals_data(data_path, fsamp)
    recording.internals_sidecar = {
        'SamplingFrequency': fsamp,
        'Description': 'Simulated internal variables: effort profile and angle profile'
    }
    
    # Write to BIDS format
    recording.write()
    
    return recording 
