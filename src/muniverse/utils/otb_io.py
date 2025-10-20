import os
import tarfile as tf
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from dateutil import parser


def open_otb(inputname, n_adapters):
    """
    Reads otb+ files and outputs stored data and metadata.
    For further details regarding the structure of the OTB+ files see:
    https://otbioelettronica.it/download/137/otb-file-structure/2665/otb-structure


    Args:
        inputname (str): name and path of the inputfile, e.g. '/this/is/mypath/filename.otb+'
        n_adapters (int): number of input adapters used in the measurement

    Returns:
        data (ndarray): array of recorded data (samples x channels)
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

    nADbit = int(device_info["ad_bits"])
    nchans = int(device_info["DeviceTotalChannels"])
    # read in the EMG trial data
    emg_data = np.fromfile(open(trial_label_sig), dtype="int" + str(nADbit))
    emg_data = np.transpose(
        emg_data.reshape(int(len(emg_data) / nchans), nchans)
    )  
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
        # trial_label_sip = os.path.join(temp_dir, sip_files[i])
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

    return (data, metadata)


def format_otb_channel_metadata(data, metadata, n_adapters):
    """
    Extract channel metadata given the output of the open_otb function

    Args:
        data (ndarray): array of recorded data (samples x channels)
        metadata (dict): metadata of the recording

    Returns:
        ch_metadata (dict): metadata associated with the individual channels
    """

    # Initalize lists for each metadata field
    ch_names = ["Ch" + str(i) for i in np.arange(1, data.shape[1] + 1)]
    units = metadata["units"]
    ch_type = []
    low_cutoff = []
    high_cutoff = []
    sampling_frequency = []
    signal_electrode = []
    grid_name = []
    group = []
    reference = []
    target_muscle = []
    interelectrode_distance = []
    description = []

    electrode_idx = 0

    # Loop over all EMG channels
    for i in np.arange(n_adapters):
        channel_metadata = metadata["adapter_info"][i].findall(".//Channel")
        n_channels = int(
            metadata["adapter_info"][i + 1].attrib["ChannelStartIndex"]
        ) - int(metadata["adapter_info"][i].attrib["ChannelStartIndex"])
        for j in np.arange(n_channels):
            electrode_idx += 1
            ch_type.append("EMG")
            high_cutoff.append(int(metadata["adapter_info"][i].attrib["LowPassFilter"]))
            low_cutoff.append(
                int(metadata["adapter_info"][i].attrib["HighPassFilter"])
            )
            sampling_frequency.append(int(metadata["device_info"]["SampleFrequency"]))
            signal_electrode.append("E" + str(electrode_idx))
            grid_name.append(channel_metadata[j].attrib["ID"])
            group.append("Grid" + str(i + 1))
            reference.append("R1")
            target_muscle.append(channel_metadata[j].attrib["Muscle"])
            tmp = channel_metadata[j].attrib["Description"]
            tmp = tmp.split("Array ")[-1]
            tmp = tmp.split((" i.e.d."))[0]
            interelectrode_distance.append(tmp)
            description.append("ElectroMyoGraphy")

    # Loop over non-EMG channels
    for i in np.arange(len(metadata["aux_info"])):
        ch_type.append("MISC")
        low_cutoff.append("n/a")
        high_cutoff.append("n/a")
        sampling_frequency.append(int(metadata["aux_info"][i]["fsample"]))
        signal_electrode.append("n/a")
        grid_name.append("n/a")
        group.append("n/a")
        reference.append("n/a")
        target_muscle.append("n/a")
        interelectrode_distance.append("n/a")
        description.append(metadata["aux_info"][i]["description"])

    # Output the channel metadata as dictonary
    ch_metadata = {
        "name": ch_names,
        "type": ch_type,
        "unit": units,
        "description": description,
        "sampling_frequency": sampling_frequency,
        "signal_electrode": signal_electrode,
        "reference_electrode": reference,
        "group": group,
        "target_muscle": target_muscle,
        "interelectrode_distance": interelectrode_distance,
        "grid_name": grid_name,
        "low_cutoff": low_cutoff,
        "high_cutoff": high_cutoff,
    }

    return ch_metadata


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
        age -= 1

    # Create dictonary with subject metadata
    subject = {}
    subject["name"] = sub_id
    subject["age"] = age
    subject["sex"] = metadata["subject_info"]["sex"]
    subject["hand"] = "n/a"
    subject["weight"] = metadata["subject_info"]["weight"]
    subject["height"] = metadata["subject_info"]["height"]

    return subject
