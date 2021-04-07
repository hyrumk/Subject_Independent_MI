import numpy as np
import torch
from torch import nn, cat
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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

        :param input_list: (list[Tensor]) a list of spectral input tensors
        :return:
        '''

        concat_fusion = cat([cnn(x) for x,cnn in zip(input_list,self.features)], dim = 0)
        output = self.fc_module(concat_fusion)
        return output



#def spectral_input_to_dataloader():

def numpy_to_trainloader(X,Y, batch_size, num_workers = 4):
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(Y)
    dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)



def train(epoch, train_loader, model, optimizer, batch_size = 100):

    use_cuda = torch.cuda.is_available()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad() # set the gradients to zero before starting backpropragation
        output = model(data) # model prediction
        loss = F.nll_loss(output, target) # calculate negative log likelihood
        loss.backward() # calculate gradients
        optimizer.step() # update gradients
        if batch_idx % 100 == 0: # print train loss per every 100 batch_idx.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_and_test(train_loader, epoch, test_loader):
    use_cuda = torch.cuda.is_available()
    model = SpectralNet(20).cuda() if use_cuda else SpectralNet(20)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    train(epoch, train_loader, model, optimizer)

# <TODO> finish train, train_and_test, numpy_to_trainloader



#summary(SpectralNet, (28,28,20))
model = SpectralNet(20).cuda()
model1 = SpectralNet(20)
print(model1)
