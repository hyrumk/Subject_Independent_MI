from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess, exponential_moving_standardize
from braindecode.datasets import MOABBDataset
import numpy as np
from braindecode.datautil.windowers import create_windows_from_events

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

def preprocess_data(dataset, low_cut_hz = 4,
                            high_cut_hz = 38,
                            factor_new = 1e-3,
                            init_block_size = 1000):
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
    # Transform the data
    preprocess(dataset, preprocessors)

    return dataset

def create_windows_dataset(dataset, trial_start_offset_seconds = -0.5,
                                    trial_stop_offset_samples=0):
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
    )

    return windows_dataset


'''
If this kind of error occurs:
ImportError: cannot import name '_check_windowing_arguments' from 'braindecode.datautil.windowers' (/home/parietal/sfreybur/miniconda3/lib/python3.7/site-packages/braindecode/datautil/windowers.py)

Refer to:
pip install from github link (https://github.com/braindecode/braindecode)
'''





