

def emg_sidecar_template(ID):
    emg_sidecar = {}
    if ID == 'Caillet2023':
        emg_sidecar['EMGPlacementScheme'] = ('Four grids were carefully positioned side-to-side with a 4-mm distance' + 
        'between the electrodes at the edges of adjacent grids. The 256 electrodes were centered to the muscle belly' +
        '(right tibialis anterior) and laid  within the muscle perimeter identified through palpation.')
        emg_sidecar['EMGElectrodeGroups'] = {
            'Grid1': {'ElectrodeManufacturer': 'OTBioelettronica', 'ElectrodeManufaturerModelName': 'GR04MM1305', 'InterelectrodeDistance': '4 mm', 'GridShape': [13, 5], 'Material': 'gold coated'},
            'Grid2': {'ElectrodeManufacturer': 'OTBioelettronica', 'ElectrodeManufaturerModelName': 'GR04MM1305', 'InterelectrodeDistance': '4 mm', 'GridShape': [13, 5], 'Material': 'gold coated'},
            'Grid3': {'ElectrodeManufacturer': 'OTBioelettronica', 'ElectrodeManufaturerModelName': 'GR04MM1305', 'InterelectrodeDistance': '4 mm', 'GridShape': [13, 5], 'Material': 'gold coated'},
            'Grid4': {'ElectrodeManufacturer': 'OTBioelettronica', 'ElectrodeManufaturerModelName': 'GR04MM1305', 'InterelectrodeDistance': '4 mm', 'GridShape': [13, 5], 'Material': 'gold coated'}
        }
        emg_sidecar['EMGReference'] = 'Wet reference band placed above the medial malleolus of the same leg.'
        emg_sidecar['EMGChannelCount'] = 256
        emg_sidecar['EMGElectrodeCount'] = 258
        emg_sidecar['SkinPreparation'] = 'The skin was shaved, abrased and cleansed with 70 percent ethyl alcohol'
        emg_sidecar['Manufacturer'] = 'OTBioelettronica'
        emg_sidecar['ManufacturerModelName'] = 'Quattrocento'
        emg_sidecar['SoftwareVersions'] = 'OTBioLab+ v.1.5.6.0'
        emg_sidecar['SamplingFrequency'] = 2048
        emg_sidecar['PowerLineFrequency'] = 50
        emg_sidecar['SoftwareFilters'] = {'bandpass filter': {'highpass': 10, 'lowpass': 500}}
        emg_sidecar['SetUpDescription'] = ('The participant sat on a massage table with the hips' +  
        'flexed at 30 degree, 0 degree being the hip neutral position, and their knees fully extended.' + 
        'We fixed the foot of the dominant leg (right in all participants) onto the pedal of a commercial' + 
        'dynamometer (OT Bioelettronica) positioned at 30 degree in the plantarflexion direction,' + 
        '0 degree being the foot perpendicular to the shank. The thigh was fixed to the massage table' + 
        'with an inextensible 3-cm-wide Velcro strap. The foot was fixed to the pedal with inextensible' + 
        'straps positioned around the proximal phalanx, metatarsal and cuneiform. Force signals were' + 
        'recorded with a load cell (CCT Transducer s.a.s.) connected in-series to the pedal using the same' + 
        'acquisition system as for the HD-EMG recordings (EMG-Quattrocento; OT Bioelettronica).' + 
        'The dynamometer was positioned accordingly to the participant’s lower limb length and' + 
        'secured to the massage table to avoid any motion during the contractions.')
        emg_sidecar['TaskName'] = 'Isometric-ankle-dorsiflexion'
        emg_sidecar['TaskDescription'] = ('Each participant performed two trapezoidal contractions at 30 percent' +
                                           'and 50 percent MVC with 120 s of rest in between, consisting of linear' +
                                            ' ramps up and down performed at 5 percent per second and aplateau ' +
                                            'maintained for 20 and 15 s at 30 percent and 50 percent MVC, respectively.' +
                                             ' The order of the contractions was randomized.')
        emg_sidecar['Instructions'] = 'Follow path provided via visual feedback.'
        emg_sidecar['SubjectPosition'] = 'seated'
        emg_sidecar['HipFlexion'] = '30 degree'
        emg_sidecar['AnkleFlexion'] = '30 degree'
        emg_sidecar['MISCChannelCount'] = 3
        emg_sidecar['MISCChannelDescription'] = {
            '1': 'Voltage output of the dynamometer load cell',
            '2': 'Requested ankle torque trajectory',
            '3': 'Performed ankle torque trajectory'
        }
        emg_sidecar['InstitutionName'] = 'Imperial Collage London'
        emg_sidecar['InstitutionAddress'] = 'London SW7 2AZ, United Kingdom'
        emg_sidecar['InstitutionalDepartmentName'] = 'Department of Bioengineering'
    else:
        # dictonary with essential fields     
        emg_sidecar['EMGPlacementScheme'] = 'other'
        emg_sidecar['EMGReference'] = 'n/a'
        emg_sidecar['SamplingFrequency'] = int()
        emg_sidecar['PowerLineFrequency'] = float()
        emg_sidecar['SoftwareFilters'] = 'n/a'

    return(emg_sidecar)

def dataset_sidecar_template(ID):

    dataset_sidecar = {}
    if ID == 'Caillet2023':
        dataset_sidecar['Name'] = 'Caillet_et_al_2023_eNeuro_BIDS_MUniverse' 
        #dataset_sidecar['BIDSversion'] = 'unpublished emg proposal'
        dataset_sidecar['DatasetType']  = 'raw' 
        dataset_sidecar['License'] = 'CC0 BY 4.0'
        dataset_sidecar['Authors'] = ['Arnault H. Caillet', 'Simon Avrillon', 'Aritra Kundu', 
                                       'Tianyi Yu', 'Andrew T. M. Phillips', 'Luca Modenese',
                                       'Dario Farina']
        dataset_sidecar['EthicsApprovals'] = ('The Ethics Committee at Imperial College London reviewed ' +
                                              'and approved all procedures and protocols (no. 18IC4685)')
        dataset_sidecar['Funding'] = [
            'European Research Council Synergy Grant NaturalBionicS (Contract #810346)',
            'Engineering and Physical Sciences Research Council (EPSRC) Transformative Healthcare', 
            'Non-Invasive Single Neuron Electrical Monitoring (NISNEM Technology) Grant EP/T020970',
            'Biotechnology and Biological Sciences Research Council (BBSRC) “Neural Commands for Fast Movements in the Primate Motor System” Grant NU-003743'
        ]
        dataset_sidecar['ReferencesAndLinks'] = [
            ('Caillet, A. H., Avrillon, S., Kundu, A., Yu, T., Phillips, A. T. M., Modenese, L., & Farina, D. (2023). ' +
             'Larger and Denser: An Optimal Design for Surface Grids of EMG Electrodes to Identify Greater and More Representative Samples of Motor Units. ' +
             'eNeuro, 10(9), ENEURO.0064-23.2023.'),
            'https://doi.org/10.1523/ENEURO.0064-23.2023',
        ]
        dataset_sidecar['GeneratedBy'] = [
            {'Name': 'muniverse', 'Version': '0.1.0', 'CodeURL': 'https://github.com/xxx'},
            {'Name': 'caillet2023_to_bids', 'Description': 'Semi-automated conversion of the original data into BIDS format'}
        ]
        dataset_sidecar['SourceDatasets'] = [{'URL': 'https://doi.org/10.6084/m9.figshare.22149287', 'Version': '2023-08-08'}]
    else:     
        dataset_sidecar['Name'] = 'n/a'
        dataset_sidecar['BIDSversion'] = 'n/a'

    return(dataset_sidecar)