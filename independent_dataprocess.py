from braindecode.datasets import MOABBDataset
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess
import copy
import numpy as np
from mne.decoding import CSP
from braindecode.datautil.windowers import create_windows_from_events
import mne
from various_computation import computeMI

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
    '''

    :param datasets: (list) a list of MOABBDatasets by subject
    :param filter_range: tuple (low_cut, high_cut)
    :return: (list) (BaseConcatDataset) a list of bandpass filtered subject data
    '''
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
    # apply bandpass filter for each subject
    for ds in datasets:
        ds_copy = copy.deepcopy(ds)
        preprocess(ds_copy, preprocessors)
        trial_start_offset_seconds = -0.5

        # Extract sampling frequency, check that they are same in all datasets
        sfreq = ds_copy.datasets[0].raw.info['sfreq']
        assert all([ds_subj.raw.info['sfreq'] == sfreq for ds_subj in ds_copy.datasets])
        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

        # Create windows using braindecode function for this. It needs parameters to define how
        # trials should be used.

        windows_dataset = create_windows_from_events(
            ds_copy,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=True,
        )

        filtered_ds.append(windows_dataset)

    return filtered_ds

def windows_to_XY(windows_dataset):
    '''
    Convert windows_dataset suitable to MNE by converting into numpy

    :param windows_dataset: (windows_dataset) from one element of returned data of bandpass_data
    :return: (numpy, numpy) X, Y
    '''
    Y = np.array([])
    for i, (x, y, window_ind) in enumerate(windows_dataset):
        if i == 0:
            X = np.array([x])
            Y = np.append(Y, y)
        else:
            X = np.concatenate([X, [x]])
            Y = np.append(Y, y)

    X = np.float64(X)
    Y = np.int64(Y)

    return X,Y


def generate_ss_feature(dataset, num_selected_bands=20):
    '''

    :param dataset: (list) a list of MOABBDatasets by subject (directly from return of call_data)
    :param num_channels:
    :param num_selected_bands:
    :return:
    '''
    filtered_dataset = []   # will be a list of (mutual_info, filtered data -> (X,Y) from the for loop, frequency band)
    num_trials = num_subjects = N_COMPONENTS = 0

    for k,filter_range in enumerate(frequency_bands):
        kth_filtered = bandpass_data(dataset, filter_range) # [window_subj1, window_subj2 ... window_subjN]
                                                            # Each window_subj1 consisted of (x,y,window_ind) by trial
        #num_trials =  len(kth_filtered[0])# trial/subject
        #num_subjects = len(kth_filtered)
        #trial_time_sample = kth_filtered[0][0][0].shape[1]
        U = num_selected_bands//2

        list_XY = [windows_to_XY(subj_windows) for subj_windows in kth_filtered]
        list_X = [subj_data[0] for subj_data in list_XY]
        list_Y = [subj_data[1] for subj_data in list_XY]

        #<TODO> possible necessary change in X shape currently (data #, channel, time point) -> (data #, time point, channel) CHANGE IT IF RESULT GETS WEIRD
        X = np.concatenate(list_X)  # concatenated X from all the subjects bandpass filtered by current filter range
        Y = np.concatenate(list_Y)  # concatenated X from all the subjects bandpass filtered by current filter range


        N_COMPONENTS = len(np.unique(Y)) if N_COMPONENTS == 0 else N_COMPONENTS
        csp = CSP(n_components=N_COMPONENTS, reg=None, log=True, norm_trace=False)
        csp_fit = csp.fit_transform(X, Y)

        # <TODO> MIGHT NEED TO COMPUTE MUTUAL INFO ON OUR OWN
        W = csp.filters_
        Wk = np.column_stack((W[:,:U], W[:,-U:])) # first U and last U columns from the filter (U = num_selected_bands/2)
        Vk = [] # where v will be stored

        for x in X: # X.shape[0] == num_subjects*num_trials
            mat = Wk.T @ x
            v = np.log(np.var(mat))
            Vk.append(v)

        Vk = np.array(Vk)
        mutual_info = computeMI(Vk, Y)

        filtered_dataset.append((mutual_info, (X,Y), filter_range))

    return filtered_dataset
        #<TODO>


DATASET_NAME = "BNCI2014001"
data = call_data(DATASET_NAME, [1,2,3])#[i for i in range(1,10)])
filt_dataset = generate_ss_feature(data)

print(len(filt_dataset)) # should have 30
print(len(filt_dataset[0]))
print(filt_dataset[0][0])
print("mutual info list:  ",[tup[0] for tup in filt_dataset])
print("X,Y shape list: ", [(tup[1][0].shape, tup[1][1].shape) for tup in filt_dataset])
print("filter range list: ", [tup[2] for tup in filt_dataset])




