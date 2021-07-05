import numpy as np
from color import color
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
        output_dim = 4
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

def numpy_to_trainloader(X,Y, batch_size, num_workers = 2, shuffle = False):
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


'''
still working on model_train function.
'''
def model_train(model, train_loader, optimizer, loss_function, epochs = 150, valid_loader = None, lr = 0.01, weight_decay = 0.5*0.001, PATH = './independent_model.pth', device = 'gpu'):
    use_cuda = torch.cuda.is_available()
    if device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_function.to(device)
    model.to(device)


    train_acc_list, valid_acc_list, train_loss_list, valid_loss_list = [], [], [], []


    print("Epoch\tTrain loss\ttrain_accuracy\tValid loss\tvalid_accuracy")
    for epoch in range(epochs):
        train_loss = 0.0
        train_acc = 0
        total, num_correct = 0, 0  # for accuracy computation
        for batch, data in enumerate(train_loader, 0):
            model.train()

            inputs, labels = data

            # to either cpu or cuda
            inputs, labels = inputs.to(device), labels.to(device=device, dtype=torch.int64)

            # splits the tensor suitable for model input
            model_input = torch.split(inputs, 1, dim=1)

            # convert to list
            model_input = list(model_input)

            optimizer.zero_grad()

            # LOSS FUNCTION
            outputs = model(model_input)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += outputs.shape[0] * loss.item()

            # Compute accuracy

            outputs = model(model_input)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            num_correct += (predicted == labels).sum().item()

        valid_total, valid_correct = 0, 0
        valid_acc = 0
        valid_loss = 0.0
        for batch, data in enumerate(valid_loader, 0):
            inputs, labels = data

            # to either cpu or cuda
            inputs, labels = inputs.to(device), labels.to(device=device, dtype=torch.int64)

            # splits the tensor suitable for model input
            model_input = torch.split(inputs, 1, dim=1)

            # convert to list
            model_input = list(model_input)

            outputs = model(model_input)
            loss = loss_function(outputs, labels)
            valid_loss += outputs.shape[0] * loss.item()

            _, predicted = torch.max(outputs.data, 1)

            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

        train_acc = (num_correct / total) * 100
        valid_acc = (valid_correct / valid_total) * 100

        # elements to print
        train_loss_print = round(train_loss, 2) if (
                    len(train_loss_list) > 0 and train_loss >= min(train_loss_list)) else (color.GREEN + str(round(train_loss,2)) + color.END)# colored(round(train_loss, 2),'green')
        train_acc_print = round(train_acc, 2) if (
                    len(train_acc_list) > 0 and train_acc <= max(train_acc_list)) else (color.RED + str(round(train_acc,2)) + color.END)
        valid_loss_print = round(valid_loss, 2) if (
                    len(valid_loss_list) > 0 and valid_loss >= min(valid_loss_list)) else (color.BLUE + str(round(valid_loss,2)) + color.END)
        valid_acc_print = round(valid_acc, 2) if (
                    len(valid_acc_list) > 0 and valid_acc <= max(valid_acc_list)) else (color.CYAN + str(round(valid_acc,2)) + color.END)

        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        print("{}\t{}\t\t{}%\t\t{}\t\t{}%".format(epoch, train_loss_print, train_acc_print, valid_loss_print,
                                                  valid_acc_print))


def train(epochs, train_loader, model, optimizer, loss_function, PATH = './independent_model.pth', device = 'gpu'):
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
    if device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        running_loss = 0.0
        train_acc = 0
        for batch, data in enumerate(train_loader,0):
            model.train()

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

            epoch_loss += outputs.shape[0] * loss.item()
            running_loss += loss.item()

            if batch % 100 == 0:
                #<TODO> saves the model with the minimum loss
                '''
                print(outputs.shape)
                print(outputs)
                print(torch.sum(torch.sum(outputs, dim=1)))
                print(labels)
                '''
                running_loss = 0.0
            model.eval()
            prediction = model(model_input)
            #train_acc += (prediction == labels).float().sum()
        print("Training Epoch {} Loss: ".format(epoch), running_loss)
    print("Finished Training")
    torch.save(model.state_dict(),PATH)

    return model


def train_and_test(train_loader, epoch, test_loader, num_classes = 4, num_band = 20, PATH = './independent_model.pth', device = 'gpu'):
    use_cuda = torch.cuda.is_available()
    if device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SpectralNet(num_band, output_dim=num_classes).cuda() if use_cuda else SpectralNet(num_band,  output_dim=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    loss_function = nn.CrossEntropyLoss().to(device)

    trained_model = train(epoch, train_loader, model, optimizer, loss_function, PATH, device)
    #test_model = SpectralNet(20).to(device)
    #test_model.load_state_dict(torch.load(PATH))
    #test_model.eval()
    trained_model.eval()
    test_model = trained_model
    correct = 0
    total = 0
    class_total = list(0. for i in range(num_classes))
    class_correct = list(0. for i in range(num_classes))
    pred_total = list(0. for i in range(num_classes))
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
            #<TODO> Fix how you come up with accuracy
            for i,pred in enumerate(predicted):
                answer = labels[i]
                class_total[answer] += 1 if labels[i] == answer else 0
                pred_total[pred] += 1
                class_correct[pred] += 1 if pred == answer else 0
                correct += 1 if pred == answer else 0

    result = "RESULT\n\n"
    result += "Total Accuracy: {}\n".format(float(correct/total))
    result += "=== Precision ===\n"
    result += ''.join(["Class {}: {}\n".format(i, float(class_correct[i]/pred_total[i]))for i in range(len(class_total))])
    '''
    result += "=== Recall ===\n"
    result += ''.join(["Class {}: {}\n".format(i, float(class_correct[i]/class_total[i]))for i in range(len(class_total))])
    '''
    print(result)

    return result

# <TODO> finish train, train_and_test, numpy_to_trainloader



#summary(SpectralNet, (28,28,20))
