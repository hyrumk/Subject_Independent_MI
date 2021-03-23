from braindecode.datasets import MOABBDataset
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess
import copy

frequency_bands = [(7.5,14),(11,13),(10,14),(9,12),(19,22),(16,22),(26,34),(17.5,20.5),(7,30),
                   (5,14),(11,31),(12,18),(7,9),(15,17),(25,30),(20,25),(5,10),(10,25),(15,30),
                   (10,12),(23,27),(28,32),(12,33),(11,22),(5,8),(7.5,17.5),(23,26),(5,20),(5,25),(10,20)]

def call_data(dataset_name, subject_ids):
    '''

    :param dataset_name: (String) data name to be entered in MOABBDataset
                        (e.g. "BNCI2014001", "BNCI2014004")
    :param subject_ids: (list) A list of subject numbers to call
            BNCI2014001: 1~9
            BNCI2014004: 1~9
    :return: (list) a list of braindecode.datasets.moabb.MOABBDataset
    '''
    datasets = [MOABBDataset(dataset_name=dataset_name, subject_ids=[i]) for i in subject_ids]
    return datasets

def bandpass_data(datasets, filter_range):
    low_cut_hz = filter_range[0]  # low cut frequency for filtering
    high_cut_hz = filter_range[1]  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    preprocessors = [
        # keep only EEG sensors
        MNEPreproc(fn='pick_types', eeg=True, meg=False, stim=False),
        # convert from volt to microvolt, directly modifying the numpy array
        NumpyPreproc(fn=lambda x: x * 1e6),
        # bandpass filter
        MNEPreproc(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
        # exponential moving standardization
        NumpyPreproc(fn=exponential_moving_standardize, factor_new=factor_new,
                     init_block_size=init_block_size)
    ]

    filtered_ds = []
    for ds in datasets:
        ds_copy = copy.deepcopy(ds)
        preprocess(ds_copy, preprocessors)
        filtered_ds.append(ds_copy)




