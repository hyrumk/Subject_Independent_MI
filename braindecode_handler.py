import numpy as np
import torch
from torch import nn, cat
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader




def windows_to_XY(windows_dataset):
    '''
    Converts windows_dataset suitable to MNE by converting into numpy

    :param windows_dataset: (windows_dataset) from windows_dataset created from create_windows_from_events function
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

def extract_channel_data(data, channel):
  '''
  data: (numpy array) numpy of shape (channel, time points)
  channel: (int) channel number (from 0 ~ len(data)-1)

  return: (list) 1-D EEG data in a list form

  '''
  EEG = list(data[channel])
  return EEG



def prepare_selected_channel_data(data_list, channel_list):
  '''
  prepares the data of a single trial including data from channels in the channel_list

  return: (list[numpy array])

  '''
  selected_data_list = []
  for data in data_list:
    trial_data = [extract_channel_data(data, channel) for channel in channel_list]
    trial_data = np.array(trial_data)
    #trial_data = trial_data.reshape((1, trial_data.shape[1], trial_data.shape[0]))
    selected_data_list.append(trial_data)
  return selected_data_list


def trial_to_DataLoader(TrialList, Labels):
  '''
  trial data list from prepare_selected_channel_data to Dataset, DataLoader
  '''
  tensor_x = torch.Tensor(TrialList) # transform to torch tensor
  tensor_y = torch.Tensor(Labels)

  my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
  my_dataloader = DataLoader(my_dataset) # create your dataloader

  return my_dataset, my_dataloader

