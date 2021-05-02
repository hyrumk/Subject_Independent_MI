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
#from torchsummary import summary


class SpectralNet(nn.Module):
    def __init__(
        self,
        n_selected, # number of selected bands
        #in_chans,
        output_dim = 4,
        n_classes = 2,
    ):
        super(SpectralNet, self).__init__()
        self.features = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1,10, kernel_size = 3, padding = 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(10, 14, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(14, 18, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(28*28*18, 256)
        ) for _ in range(20)])
        self.fc_module = nn.Sequential(
            nn.Linear(256*n_selected, 1024),
            nn.Linear(1024, output_dim),
        )

    def forward(self, input_list):
        '''

        :param input_list: (list[Tensor]) a list of spectral input tensors
        :return:
        '''
        #print("INPUT: ", input_list.shape)
        concat_fusion = cat([cnn(x) for x,cnn in zip(input_list,self.features)], dim = 1)
        output = self.fc_module(concat_fusion)
        return output



#def spectral_input_to_dataloader():

def numpy_to_trainloader(X,Y, batch_size, num_workers = 4, shuffle = False):
    '''

    :param X: (numpy)
    :param Y: (numpy)
    :param batch_size: batch size you want for the trainloader
    :param num_workers:
    :param shuffle: whether you want to shuffle the data or not (don't shuffle it)
    :return: train_loader
    '''
    tensor_x = torch.FloatTensor(X)
    tensor_y = torch.Tensor(Y)
    dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers)
    return train_loader



def train(epochs, train_loader, model, optimizer, loss_function, PATH = './independent_model.pth'):
    '''

    :param epochs: (int) number of epochs to train
    :param train_loader: train_loader
    :param model: neural network model to be used in the training
    :param optimizer: the type of optimizer to be used
    :param loss_function: loss function to be used
    :param PATH:
    :return:
    '''
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch, data in enumerate(train_loader,0):
            #<TODO> Delete this after using

            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device=device, dtype=torch.int64)
            # splits the tensor suitable for model input
            model_input = torch.split(inputs, 1, dim=1)

            # convert to list
            model_input = list(model_input)
            optimizer.zero_grad()

            #outputs = torch.LongTensor(outputs.to('cpu'))
            # LOSS FUNCTION
            outputs = model(model_input)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch % 100 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch + 1, running_loss/100))
                '''
                print(outputs.shape)
                print(outputs)
                print(torch.sum(torch.sum(outputs, dim=1)))
                print(labels)
                '''
                running_loss = 0.0
    print("Finished Training")
    torch.save(model.state_dict(),PATH)

    return model


def train_and_test(train_loader, epoch, test_loader, num_band = 20, PATH = './independent_model.pth'):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SpectralNet(num_band).cuda() if use_cuda else SpectralNet(num_band)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    loss_function = nn.CrossEntropyLoss().to(device)

    trained_model = train(epoch, train_loader, model, optimizer, loss_function, PATH)
    #test_model = SpectralNet(20).to(device)
    #test_model.load_state_dict(torch.load(PATH))
    #test_model.eval()
    trained_model.eval()
    test_model = trained_model
    correct = 0
    total = 0
    class_total = list(0. for i in range(4))
    class_correct = list(0. for i in range(4))
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device=device, dtype=torch.int64)
            # splits the tensor suitable for model input
            model_input = torch.split(inputs, 1, dim=1)

            outputs = test_model(model_input)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            for i,pred in enumerate(predicted):
                if labels[i] == 0:
                    class_total[0] += 1
                elif labels[i] == 1:
                    class_total[1] += 1
                elif labels[i] == 2:
                    class_total[2] += 1
                else:
                    class_total[3] += 1
                if pred == labels[i]:
                    correct += 1
                    if pred == 0:
                        class_correct[0] += 1
                    elif pred == 1:
                        class_correct[1] += 1
                    elif pred == 2:
                        class_correct[2] += 1
                    else:
                        class_correct[3] += 1

    print("RESULT\n\n")
    print("Total: ", float(correct/total))
    print("Class 0: ", float(class_correct[0]/class_total[0]))
    print("Class 1: ", float(class_correct[1] / class_total[1]))
    print("Class 2: ", float(class_correct[2] / class_total[2]))
    print("Class 3: ", float(class_correct[3] / class_total[3]))
# <TODO> finish train, train_and_test, numpy_to_trainloader



#summary(SpectralNet, (28,28,20))
model = SpectralNet(20).cuda()
model1 = SpectralNet(20)
print(model1)
