import numpy as np
import torch
from torch import nn, cat
from torch.nn import init

from braindecode.util import np_to_var
from braindecode.models.modules import Expression, Ensure4d
from braindecode.models.functions import (
    safe_log, square, transpose_time_to_spat, squeeze_final_output
)
from torchsummary import summary


class SpectralNet(nn.Module):
    def __init__(
        self,
        n_selected, # number of selected bands
        #in_chans,
        n_classes = 2,
        cnn_output_dim = 1024
    ):
        super(SpectralNet, self).__init__()

        self.deep_cnn_list = [
            nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(10, 14, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(14, 18, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(28 * 28 * 18, 256)
            )
        ]*20

        self.features = nn.Sequential(
            nn.Conv2d(1,10, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(10, 14, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(14, 18, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(28*28*18, 256)
        )
        self.fc_module = nn.Sequential(
            nn.Linear(256*n_selected, cnn_output_dim),
            nn.Softmax(dim=n_classes)
        )

    def forward(self, input_list):
        '''

        :param input_list: (list) a list of spectral input tensors
        :return:
        '''

        concat_fusion = cat([cnn(x) for x,cnn in zip(input_list,self.features)])
        output = self.fc_module(concat_fusion)
        return output


#summary(SpectralNet, (28,28,20))
model = SpectralNet(20).cuda()
model1 = SpectralNet(20)
print(model1)
