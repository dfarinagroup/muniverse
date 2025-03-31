# This script is copied from NeuroMotion/scripts/mov2emg.py

import argparse
import os
import torch
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from easydict import EasyDict as edict
from scipy.signal import butter, filtfilt
import yaml

import sys
sys.path.append('.')

from NeuroMotion.MSKlib.MSKpose import MSKModel
from NeuroMotion.MNPoollib.MNPool import MotoneuronPool
from NeuroMotion.MNPoollib.mn_utils import generate_emg_mu, normalise_properties
from NeuroMotion.MNPoollib.mn_params import DEPTH, ANGLE, MS_AREA, NUM_MUS, mn_default_settings
from BioMime.models.generator import Generator
from BioMime.utils.basics import update_config, load_generator


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate EMG signals from movements')
    parser.add_argument('config_path', type=str, help='Path to input configuration YAML file')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config_path)
    
    # Set default values from config or use defaults
    model_pth = cfg.get('model_pth', './ckp/model_linear.pth')
    model_config = update_config('./ckp/config.yaml')
    device = cfg.get('device', 'cpu')
    morph = cfg.get('morph', False)
    muap_file = cfg.get('muap_file', './ckp/muap_examples.pkl')
    
    # PART ONE: Define MSK model, movements, and extract param changes
    msk = MSKModel(
        model_path = './NeuroMotion/MSKlib/models/ARMS_Wrist_Hand_Model_4.3/',
        model_name= 'Hand_Wrist_Model_for_development.osim',
        default_pose_path = './NeuroMotion/MSKlib/models/poses.csv',
    )
        
    poses = ['default', 'default+flex', 'default', 'default+ext', 'default']
    durations = [1.5] * 4
    duration = np.sum(durations)
    fs_mov = 5
    msk.sim_mov(fs_mov, poses, durations)

    # # Load joint angles from file
    # file_path = './data/joint_angle.pkl'
    # with open(file_path, 'rb') as file:
    #     joint_angles = pickle.load(file)        # pd.dataframe or np.array
    # duration = 10       # seconds
    # fs_mov = 5
    # msk.load_mov(joint_angles)

    ms_labels = ['ECRB', 'ECRL', 'PL', 'FCU', 'ECU', 'EDCI', 'FDSI']
    ms_lens = msk.mov2len(ms_labels=ms_labels)
    changes = msk.len2params()
    steps = changes['steps']

    # PART TWO: Define the MotoneuronPool of one muscle
    ms_label = cfg.get('target_muscle', 'FDSI')

    if morph:
        with open(muap_file, 'rb') as fl:
            db = pickle.load(fl)
        num_mus = len(db['iz'])
    else:
        num_mus = NUM_MUS[ms_label]

    mn_pool = MotoneuronPool(num_mus, ms_label, **mn_default_settings)
    # Assign physiological properties
    fibre_density = cfg.get('fibre_density', 200)     # 200 fibres per mm^2
    num_fb = np.round(MS_AREA[ms_label] * fibre_density)    # total number within one muscle
    config = edict({
        'num_fb': num_fb,
        'depth': DEPTH[ms_label],
        'angle': ANGLE[ms_label],
        'iz': [0.5, 0.1],
        'len': [1.0, 0.05],
        'cv': [4, 0.3]      # Recommend not setting std too large. cv range in training dataset is [3, 4.5]
    })

    if morph:
        num, depth, angle, iz, cv, length, base_muaps = normalise_properties(db, num_mus, steps)
    else:
        properties = mn_pool.assign_properties(config, normalise=True)
        num = torch.from_numpy(properties['num']).reshape(num_mus, 1).repeat(1, steps)
        depth = torch.from_numpy(properties['depth']).reshape(num_mus, 1).repeat(1, steps)
        angle = torch.from_numpy(properties['angle']).reshape(num_mus, 1).repeat(1, steps)
        iz = torch.from_numpy(properties['iz']).reshape(num_mus, 1).repeat(1, steps)
        cv = torch.from_numpy(properties['cv']).reshape(num_mus, 1).repeat(1, steps)
        length = torch.from_numpy(properties['len']).reshape(num_mus, 1).repeat(1, steps)

    fs = cfg.get('fs', 2048)
    mn_pool.init_twitches(fs)
    mn_pool.init_quisistatic_ef_model()
    ext = np.concatenate((np.linspace(0, 0.8, round(fs * duration / 2)), np.linspace(0.8, 0, round(fs * duration / 2))))      # percentage MVC
    time_samples = len(ext)
    ext_new, spikes, fr, ipis = mn_pool.generate_spike_trains(ext, fit=False)

    # PART THREE: Simulate MUAPs using BioMime during the movement
    if ms_label == 'FCU_u' or ms_label == 'FCU_h':
        tgt_ms_labels = ['FCU'] * num_mus
    else:
        tgt_ms_labels = [ms_label] * num_mus

    ch_depth = changes['depth'].loc[:, tgt_ms_labels]
    ch_cv = changes['cv'].loc[:, tgt_ms_labels]
    ch_len = changes['len'].loc[:, tgt_ms_labels]

    # Model
    generator = Generator(model_config.Model.Generator)
    generator = load_generator(model_pth, generator, device)
    generator.eval()

    # Device
    if device == 'cuda':
        assert torch.cuda.is_available()
        generator.cuda()

    # Filtering parameters from config
    time_length = cfg.get('time_length', 96)
    cutoff = cfg.get('cutoff', 800)
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    order = cfg.get('filter_order', 4)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    start_time = time.time()

    muaps = []
    for sp in tqdm(range(steps), dynamic_ncols=True, desc='Simulating MUAPs during dynamic movement...'):
        cond = torch.vstack((
            num[:, sp],
            depth[:, sp] * ch_depth.iloc[sp, :].values,
            angle[:, sp],
            iz[:, sp],
            cv[:, sp] * ch_cv.iloc[sp, :].values,
            length[:, sp] * ch_len.iloc[sp, :].values,
        )).transpose(1, 0)

        if not morph:
            zi = torch.randn(num_mus, model_config.Model.Generator.Latent)
            if device == 'cuda':
                zi = zi.cuda()
        else:
            if device == 'cuda':
                base_muaps = base_muaps.cuda()

        if device == 'cuda':
            cond = cond.cuda()

        if morph:
            sim = generator.generate(base_muaps, cond.float())
        else:
            sim = generator.sample(num_mus, cond.float(), cond.device, zi)

        if device == 'cuda':
            sim = sim.permute(0, 2, 3, 1).cpu().detach().numpy()
        else:
            sim = sim.permute(0, 2, 3, 1).detach().numpy()

        num_mu_dim, n_row_dim, n_col_dim, n_time_dim = sim.shape
        sim = filtfilt(b, a, sim.reshape(-1, n_time_dim))
        muaps.append(sim.reshape(num_mu_dim, n_row_dim, n_col_dim, n_time_dim).astype(np.float32))

    muaps = np.array(muaps)
    muaps = np.transpose(muaps, (1, 0, 2, 3, 4))
    print('--- %s seconds ---' % (time.time() - start_time))

    # PART FOUR: generate EMG signals
    _, _, n_row, n_col, time_length = muaps.shape
    emg = np.zeros((n_row, n_col, time_samples + time_length))
    for mu in np.arange(num_mus):
        emg = emg + generate_emg_mu(muaps[mu], spikes[mu], time_samples)

    print('All done, emg.shape: ', emg.shape)
    
    # Create output directory if it doesn't exist
    # os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the EMG data
    output_path = os.path.join(args.output_dir, 'emg.npy')
    np.save(output_path, emg)
    print(f'EMG data saved to {output_path}')