import os, re
import shutil
import tarfile as tf
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser


def _open_otb(inputname, n_adapters=None):
    """
    Reads otb+ files and outputs stored data and metadata.
    For further details regarding the structure of the OTB+ files see:
    https://otbioelettronica.it/download/137/otb-file-structure/2665/otb-structure

    File structure: an .otb+ archive is a tar containing a single <name>.sig (raw int16 binary,
    interleaved samples × channels) and a matching <name>.xml (device/adapter/channel metadata).
    Task recordings additionally include .sip/.pro file pairs: .pro holds per-signal XML metadata
    (description, fsample, units) and .sip holds the corresponding float64 signal (e.g. acquired
    force, requested path %MVC, performed path). These auxiliary signals are appended after the
    EMG channels in the returned data array.

    Args:
        inputname (str): name and path of the inputfile, e.g. '/this/is/mypath/filename.otb+'
        n_adapters (int, optional): number of EMG input adapters. Inferred from the XML if not given.

    Returns:
        data (ndarray): array of recorded data (channels x samples)
        metadata (dict): metadata of the recording
    """

    #
    filename = inputname.split("/")[-1]
    temp_dir = os.path.join("./", "temp_tarholder")
    # make a temporary directory to store the data of the otb file if it doesn't exist yet
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    # Open the .tar file and extract all data
    with tf.open(inputname, "r") as emg_tar:
        emg_tar.extractall(temp_dir)

    # Extract file names from .tar directory
    sig_files = [f for f in os.listdir(temp_dir) if f.endswith(".sig")]
    trial_label_sig = sig_files[0]
    # only one .sig so can be used to get the trial name (0 index list->string)
    trial_label_xml = trial_label_sig.split(".")[0] + ".xml"
    trial_label_sig = os.path.join(temp_dir, trial_label_sig)
    trial_label_xml = os.path.join(temp_dir, trial_label_xml)
    sip_files = [f for f in os.listdir(temp_dir) if f.endswith(".pro")]

    # read the metadata xml file
    with open(trial_label_xml, encoding="utf-8") as file:
        xml = ET.fromstring(file.read())

    # Get the device info
    device_info = xml.attrib

    # Get the power supply (in volt), needed for the conversion to phyisical units
    if device_info["Name"].split(";")[0] == "QUATTROCENTO":
        PowerSupply = 5
    else:
        raise ValueError("Unsupported device")

    # Get the adapter info
    adapter_info = xml.findall(".//Adapter")

    # Infer n_adapters from XML if not supplied: count non-control, non-aux adapters
    if n_adapters is None:
        _non_emg = ("AdapterControl", "Direct connection to Auxiliary Input")
        n_adapters = sum(1 for a in adapter_info if a.attrib["ID"] not in _non_emg)

    nADbit = int(device_info["ad_bits"])
    nchans = int(device_info["DeviceTotalChannels"])
    # read in the EMG trial data
    emg_data = np.fromfile(open(trial_label_sig), dtype=f"int{nADbit}")
    emg_data = np.transpose(emg_data.reshape(int(len(emg_data) / nchans), nchans))
    # need to reshape because it is read as a stream
    emg_data = emg_data.astype(float)

    # initalize vector of recorded units
    ch_units = []

    # Get the number of EMG channels per adapter
    ch_per_adpaters = np.zeros(n_adapters)
    for i in range(n_adapters):
        ch_per_adpaters[i] = int(adapter_info[i + 1].attrib["ChannelStartIndex"]) - int(
            adapter_info[i].attrib["ChannelStartIndex"]
        )

    # Get the total number of EMG channels
    n_channels = int(sum(ch_per_adpaters))

    # initalize data vector
    data = np.zeros((emg_data.shape[1], n_channels + len(sip_files)))

    # convert the data from bits to microvolts
    ch_idx = 0
    for i in range(n_adapters):
        gain = float(adapter_info[i].attrib["Gain"])
        for j in range(int(ch_per_adpaters[i])):
            # The coversion formula is derived from:
            # https://github.com/OTBioelettronica/OTB-Matlab/blob/main/MATLAB%20Open%20and%20Processing%20OTBFiles/OpenOTBFiles/OpenOTBfilesConvFact.m
            data[:, ch_idx] = (np.dot(emg_data[ch_idx, :], PowerSupply * 1000)) / (
                2 ** float(nADbit) * gain
            )
            ch_units.append("mV")
            ch_idx += 1

    # Get data and metadata from the aux input channels
    aux_info = dict()

    for i in range(len(sip_files)):
        # Get metadata
        tmp = sip_files[i]
        tmp = tmp.split(".")[0] + ".pro"
        tmp = os.path.join(temp_dir, tmp)
        with open(tmp, encoding="utf-8") as file:
            xml = ET.fromstring(file.read())

        aux_info[i] = {child.tag: child.text for child in xml}
        ch_units.append(aux_info[i]["unity_of_measurement"])

        # get data
        trial_label_sip = sip_files[i]
        trial_label_sip = trial_label_sip.split(".")[0] + ".sip"
        trial_label_sip = os.path.join(temp_dir, trial_label_sip)
        # trial_label_sip = os.path.   join(temp_dir, sip_files[i])
        aux_data = np.fromfile(open(trial_label_sip), dtype="float64")
        aux_data = aux_data[0 : data.shape[0]]
        data[:, i + n_channels] = aux_data

    # Get the subject info
    with open(os.path.join(temp_dir, "patient.xml"), encoding="utf-8") as file:
        xml = ET.fromstring(file.read())

    subject_info = {child.tag: child.text for child in xml}

    # Remove .tar folder
    for filename in os.listdir(temp_dir):
        file = os.path.join(temp_dir, filename)
        if os.path.isfile(file):
            os.remove(file)

    os.rmdir(temp_dir)

    metadata = {
        "device_info": device_info,
        "adapter_info": adapter_info,
        "aux_info": aux_info,
        "subject_info": subject_info,
        "units": ch_units,
    }

    return (data.T, metadata)


def _open_otb4(inputname):
    """
    Reads otb4 files and outputs stored data and metadata.

    File structure: an .otb4 archive is a tar containing a <name>.sig (raw int16, interleaved
    samples × channels), DeviceParameters.xml (device-level: AdBits, SamplingFrequency, ADC_Range),
    and Tracks_000.xml (per-track metadata: channel layout, gain, units, filter settings).
    Task recordings additionally include a TrapezoidalTracks_*.xml and a paired float64 .sig file
    storing the performed path and original (requested) path in %MVC at 10 Hz. These are upsampled
    to the main EMG sampling rate via linear interpolation and appended as extra channels so the
    full recording is returned as a single array.

    Args:
        inputname (str): path to the .otb4 file

    Returns:
        data (ndarray): array of recorded data (channels x samples), in physical units
        metadata (dict): metadata of the recording
    """
    temp_dir = tempfile.mkdtemp()
    try:
        with tf.open(inputname, "r") as archive:
            archive.extractall(temp_dir)

        # Read device-level parameters
        with open(os.path.join(temp_dir, "DeviceParameters.xml"), encoding="utf-8") as f:
            dev_xml = ET.fromstring(f.read())

        n_ad_bits = int(dev_xml.findtext("AdBits"))
        fs = int(dev_xml.findtext("SamplingFrequency"))
        adc_range = float(dev_xml.findtext("ADC_Range"))

        # Read per-track metadata
        with open(os.path.join(temp_dir, "Tracks_000.xml"), encoding="utf-8") as f:
            tracks_xml = ET.fromstring(f.read())

        tracks = tracks_xml.findall("TrackInfo")
        total_channels = int(tracks[0].findtext("TotalChannelsInFile"))

        # Read raw binary signal: (samples x total_channels)
        sig_files = [f for f in os.listdir(temp_dir) if f.endswith(".sig")]
        raw = np.fromfile(
            os.path.join(temp_dir, sig_files[0]), dtype=np.int16
        ).reshape(-1, total_channels).astype(np.float64)

        # Build per-channel conversion factors and collect track metadata
        # Conversion: raw * (adc_range * unit_factor) / (2^n_ad_bits * gain)
        conv_factors = np.ones(total_channels)
        units = [""] * total_channels
        track_info = []

        def _parse_hz(text):
            try:
                return float(text.split()[0])
            except (AttributeError, ValueError, IndexError):
                return np.nan

        for track in tracks:
            acq_ch = int(track.findtext("AcquisitionChannel"))
            n_ch = int(track.findtext("NumberOfChannels"))
            gain = float(track.findtext("Gain"))
            unit_factor = float(track.findtext("UnitOfMeasurementFactor"))
            unit = track.findtext("UnitOfMeasurement")
            track_adc_range = float(track.findtext("ADC_Range") or adc_range)
            sd = track.find("StringsDescriptions")

            conv = (track_adc_range * unit_factor) / (2**n_ad_bits * gain) if gain != 0 else 1.0
            conv_factors[acq_ch:acq_ch + n_ch] = conv
            for i in range(acq_ch, acq_ch + n_ch):
                units[i] = unit

            track_info.append({
                "title": track.findtext("Title"),
                "subtitle": track.findtext("SubTitle") or "",
                "is_control": track.findtext("IsControl") == "true",
                "acq_channel": acq_ch,
                "n_channels": n_ch,
                "gain": gain,
                "unit": unit,
                "fs": int(track.findtext("SamplingFrequency")),
                "low_cutoff": _parse_hz(sd.findtext("HighPassFilter") if sd is not None else None),
                "high_cutoff": _parse_hz(sd.findtext("LowPassFilter") if sd is not None else None),
                "mode": (sd.findtext("Mode") if sd is not None else None) or "n/a",
                "sensor": (sd.findtext("OriginalSensor") if sd is not None else None) or "n/a",
            })

        data = (raw * conv_factors).T  # (channels x samples)

        # Discard internal device channels (ramp generators, buffers, counters),
        # mirroring the otb+ reader which only outputs EMG + processed aux channels.
        # This is done after the main data is built but before trapezoidal tracks are
        # appended so that control-channel indices don't contaminate the final array.
        control_mask = np.zeros(data.shape[0], dtype=bool)
        ch_offset = 0
        for track in track_info:
            n = track["n_channels"]
            if track["is_control"]:
                control_mask[ch_offset:ch_offset + n] = True
            ch_offset += n
        data = data[~control_mask]
        units = [u for u, skip in zip(units, control_mask) if not skip]
        track_info = [t for t in track_info if not t["is_control"]]
        total_channels = data.shape[0]  # update after filtering

        # Read trapezoidal feedback tracks if present (task recordings only)
        n_main_samples = data.shape[1]
        for trap_xml_path in sorted(Path(temp_dir).glob("TrapezoidalTracks_*.xml")):
            with open(trap_xml_path, encoding="utf-8") as f:
                trap_xml = ET.fromstring(f.read())
            trap_tracks = trap_xml.findall("TrackInfo")
            if not trap_tracks:
                continue
            sig_name = trap_tracks[0].findtext("SignalStreamPath")
            trap_total_ch = int(trap_tracks[0].findtext("TotalChannelsInFile"))
            trap_fs = int(trap_tracks[0].findtext("SamplingFrequency"))
            # SampleSize=8 → stored as float64, already in physical units (%MVC)
            trap_raw = np.fromfile(
                os.path.join(temp_dir, sig_name), dtype=np.float64
            ).reshape(-1, trap_total_ch)
            # Upsample to main fs via linear interpolation
            t_trap = np.arange(trap_raw.shape[0]) / trap_fs
            t_main = np.arange(n_main_samples) / fs
            trap_up = np.stack([
                np.interp(t_main, t_trap, trap_raw[:, ch])
                for ch in range(trap_total_ch)
            ])  # (trap_total_ch, n_main_samples)
            data = np.vstack([data, trap_up])
            for trap_track in trap_tracks:
                acq_ch_local = int(trap_track.findtext("AcquisitionChannel"))
                n_ch = int(trap_track.findtext("NumberOfChannels"))
                subtitle = trap_track.findtext("SubTitle") or ""
                unit_str = trap_track.findtext("UnitOfMeasurement") or "A.U."
                units.extend([unit_str] * n_ch)
                track_info.append({
                    "title": trap_track.findtext("Title"),
                    "subtitle": subtitle,
                    "is_control": False,
                    "acq_channel": total_channels + acq_ch_local,
                    "n_channels": n_ch,
                    "gain": float(trap_track.findtext("Gain") or 1.0),
                    "unit": unit_str,
                    "fs": fs,
                    "low_cutoff": np.nan,
                    "high_cutoff": np.nan,
                    "mode": "n/a",
                    "sensor": "n/a",
                })
            total_channels += trap_total_ch

        device_info = {
            "SamplingFrequency": fs,
            "AdBits": n_ad_bits,
            "ADC_Range": adc_range,
        }

        metadata = {
            "device_info": device_info,
            "tracks": track_info,
            "units": units,
        }

        return data, metadata

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _format_otb_channel_metadata(data, metadata, n_adapters):
    """
    Extract channel metadata given the output of the open_otb function

    Args:
        data (ndarray): array of recorded data (channels x samples)
        metadata (dict): metadata of the recording

    Returns:
        ch_metadata (dict): metadata associated with the individual channels
    """

    fsamp = int(metadata["device_info"]["SampleFrequency"])

    # Initalize lists for each metadata field
    columns = [
        "name", "type", "units", "description", "sampling_frequency",
        "signal_electrode", "reference", "group", "target_muscle",
        "interelectrode_distance", "low_cutoff", "high_cutoff"
    ]

    df = pd.DataFrame(columns=columns)

    df = df.astype({
        "name": "string", 
        "type": "string", 
        "units": "string",
        "description": "string", 
        "sampling_frequency": "float",
        "signal_electrode": "string", 
        "reference": "string",
        "group": "string", 
        "target_muscle": "string", 
        "interelectrode_distance": "float",
        "low_cutoff": "float", 
        "high_cutoff": "float"
    })

    ch_idx = 0

    # Loop over all EMG channels
    for i in np.arange(n_adapters):

        # Extract adapter specific metadata
        channel_metadata = metadata["adapter_info"][i].findall(".//Channel")
        n_channels = int(
            metadata["adapter_info"][i + 1].attrib["ChannelStartIndex"]
        ) - int(metadata["adapter_info"][i].attrib["ChannelStartIndex"])
        ied = channel_metadata[i].attrib["Description"]
        ied = int(re.search(r'(\d+)\s*mm',ied).group(1))
        low_cutoff = int(metadata["adapter_info"][i].attrib["HighPassFilter"])
        high_cutoff = int(metadata["adapter_info"][i].attrib["LowPassFilter"])

        for j in np.arange(n_channels):
            
            df.loc[len(df)] = [
                f"Ch{str(ch_idx+1).zfill(3)}",        # name
                "EMG",                                # type 
                metadata["units"][ch_idx],            # units   
                "ElectroMyoGraphy",                   # description
                fsamp,                                # sampling_frequency
                f"E{str(ch_idx+1).zfill(3)}",         # signal_electrode
                "R1",                                 # reference
                f"grid{i+1}",                         # group
                channel_metadata[j].attrib["Muscle"], # target_muscle
                ied,                                  # interelectrode_distance
                high_cutoff,                          # high_cutoff  
                low_cutoff                            # low_cutoff  
            ]

            ch_idx += 1

    # Loop over non-EMG channels
    for i in np.arange(len(metadata["aux_info"])):

        df.loc[len(df)] = [
            f"Ch{str(ch_idx+1).zfill(3)}",          # name
            "MISC",                                 # type 
            metadata["units"][ch_idx],              # units   
            metadata["aux_info"][i]["description"], # description
            fsamp,                                  # sampling_frequency
            "n/a",                                  # signal_electrode
            "n/a",                                  # reference
            "n/a",                                  # group
            "n/a",                                  # target_muscle
            "n/a",                                  # interelectrode_distance
            "n/a",                                  # high_cutoff  
            "n/a"                                   # low_cutoff  
        ]

        ch_idx += 1

    return df


def _format_otb4_channel_metadata(data, metadata):
    """
    Extract channel metadata given the output of _open_otb4.
    Returns a DataFrame with the same columns as _format_otb_channel_metadata.

    #TODO: Find if/ where target muscle lives in the xml.

    Args:
        data (ndarray): array of recorded data (channels x samples)
        metadata (dict): metadata dict returned by _open_otb4

    Returns:
        df (DataFrame): per-channel metadata
    """
    columns = [
        "name", "type", "units", "description", "sampling_frequency",
        "signal_electrode", "reference", "group", "target_muscle",
        "interelectrode_distance", "low_cutoff", "high_cutoff"
    ]

    df = pd.DataFrame(columns=columns).astype({
        "name": "string",
        "type": "string",
        "units": "string",
        "description": "string",
        "sampling_frequency": "float",
        "signal_electrode": "string",
        "reference": "string",
        "group": "string",
        "target_muscle": "string",
        "interelectrode_distance": "float",
        "low_cutoff": "float",
        "high_cutoff": "float",
    })

    ch_idx = 0
    emg_grid_count = 0

    for track in metadata["tracks"]:
        n_ch = track["n_channels"]
        subtitle = track["subtitle"]

        if not track["is_control"] and track["unit"] == "mV":
            emg_grid_count += 1
            # IED from grid name, e.g. "GR04MM1305" → 4 mm
            ied_match = re.search(r"GR(\d+)MM", subtitle)
            ied = int(ied_match.group(1)) if ied_match else np.nan

            for _ in range(n_ch):
                df.loc[len(df)] = [
                    f"Ch{str(ch_idx + 1).zfill(3)}",  # name
                    "EMG",                              # type
                    track["unit"],                      # units
                    "ElectroMyoGraphy",                 # description
                    float(track["fs"]),                 # sampling_frequency
                    f"E{str(ch_idx + 1).zfill(3)}",     # signal_electrode
                    "R1",                               # reference
                    f"grid{emg_grid_count}",            # group (indexed, not grid-type name)
                    "Not defined",                      # target_muscle
                    ied,                                # interelectrode_distance
                    track["low_cutoff"],                # low_cutoff
                    track["high_cutoff"],               # high_cutoff
                ]
                ch_idx += 1

        else:
            # AUX / control channels
            sensor = track.get("sensor", "n/a") 
            description = f"{subtitle} ({sensor})" if subtitle and sensor not in ("n/a", " - ") else (subtitle or track["title"])

            for _ in range(n_ch):
                df.loc[len(df)] = [
                    f"Ch{str(ch_idx + 1).zfill(3)}",    # name
                    "MISC",                             # type
                    track["unit"],                      # units
                    description,                        # description
                    float(track["fs"]),                 # sampling_frequency
                    "n/a",                              # signal_electrode
                    "n/a",                              # reference
                    "n/a",                              # group
                    "n/a",                              # target_muscle
                    np.nan,                             # interelectrode_distance
                    track["low_cutoff"],                # low_cutoff
                    track["high_cutoff"],               # high_cutoff
                ]
                ch_idx += 1

    return df


def read_otb(filepath):
    """
    Unified OTB file reader. Auto-detects format from extension (.otb+ or .otb4)
    and returns data with normalized channel metadata.

    Args:
        filepath (str or Path): path to an .otb+ or .otb4 file

    Returns:
        data (ndarray): recorded data, shape (channels x samples), in physical units
        channel_info (DataFrame): per-channel metadata with columns:
            name, type, units, description, sampling_frequency, signal_electrode,
            reference, group, target_muscle, interelectrode_distance, low_cutoff, high_cutoff
    """
    filepath = Path(filepath)
    ext = filepath.suffix  # ".otb+" or ".otb4"

    if ext == ".otb+":
        data, metadata = _open_otb(str(filepath))
        _non_emg = ("AdapterControl", "Direct connection to Auxiliary Input")
        n_adapters = sum(1 for a in metadata["adapter_info"] if a.attrib["ID"] not in _non_emg)
        channel_info = _format_otb_channel_metadata(data, metadata, n_adapters)
    elif ext == ".otb4":
        data, metadata = _open_otb4(str(filepath))
        channel_info = _format_otb4_channel_metadata(data, metadata)
    else:
        raise ValueError(f"Unsupported OTB format: '{ext}'. Expected .otb+ or .otb4")

    return data, channel_info


def format_subject_metadata(sub_id, metadata):
    """
    Extract subject metadata given the output of the open_otb function

    Args:
        sub_id (str): array of recorded data (samples x channels)
        metadata (dict): metadata of the recording

    Returns:
        subject (dict): subject metadata
    """

    # Calculate the subject age
    start = parser.parse(metadata["subject_info"]["birth_date"])
    end = parser.parse(metadata["subject_info"]["time"])

    age = end.year - start.year

    if (end.month, end.day) < (start.month, start.day):
        age -= np.nan

    columns = [
        "participant_id", "age", "sex", "hand", "weight", "height"
    ]

    df = pd.DataFrame(columns=columns)

    df.astype({
        "participant_id": "string",
        "age": "int",
        "sex": "string",
        "hand": "string",
        "weight": "float",
        "height": "float"
    })

    sex = metadata["subject_info"]["sex"]
    weight = metadata["subject_info"]["weight"]   
    height = metadata["subject_info"]["height"]

    df.loc[len(df)] = [
        sub_id, age, sex, "n/a", weight, height
    ]     

    return df
