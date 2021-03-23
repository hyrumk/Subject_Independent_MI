import torch
from braindecode.util import set_random_seeds
from sampleNet import SampleNet
from dataprocess import call_data, preprocess_data, create_windows_dataset
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import pandas as pd
from pathlib import Path
#<TODO> Automatically find the number of types of labels.

def run_sample_model(subject_dataset, dataset_name,
                     train_test_ratio = "default",
                     file_name = "temp",
                     learning_rate = 0.0625 * 0.01,
                     weight_decay = 0,
                     batch_size=64,
                     n_epochs = 4,
                     n_classes = 4):
    '''
    Runs the sample model for a given individual subject dataset

    :param subject_dataset: (a list of MOABBDataset)
    :param dataset_name: (String) name of the dataset
    :param train_test_ratio: (float) 0 ~ 1 train and test ratio. If default, splits in half.
    :param file_name: (String) name of the plot file to save, will also be used as plot title.
    :param subject_number: (int) Assigned number of a given subject data
    :param learning_rate: (int) learning rate
    :param weight_decay: (int) weight decay
    :param batch_size: (int) batch size
    :param n_epochs: (int) number of epochs
    :param n_classes: (int) number of label types
    :return:
    '''
    lr = learning_rate
    # For deep4 they should be:
    # lr = 1 * 0.01
    # weight_decay = 0.5 * 0.001

    ## preprocess data before giving it as an input
    dataset = preprocess_data(subject_dataset)
    windows_dataset = create_windows_dataset(dataset)

    ## Split Train/Validation(Test) sets
    # Default splits into half by session
    if train_test_ratio is "default" and dataset_name is "BNCI2014001":
        splitted = windows_dataset.split('session')
        train_set = splitted['session_T']
        valid_set = splitted['session_E']
    # else, it splits the data according to given train_test split ratio
    else:
        if train_test_ratio is "default":   # when dataset is not BNCI2014001. This is because sessions are named differently.
            train_test_ratio = 0.5
        assert ((train_test_ratio >= 0)|(train_test_ratio <= 1)), "train_test_ratio must be within 0~1 range"
        number_of_runs = windows_dataset.description.shape[0]
        run_list = [i for i in range(number_of_runs)]
        random.shuffle(run_list)
        split_point = int(number_of_runs*train_test_ratio)
        splitted = windows_dataset.split([run_list[:split_point],run_list[split_point:]])
        train_set, valid_set = splitted["0"], splitted["1"]


    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'

    if cuda:
        torch.backends.cudnn.benchmark = True
    seed = 20200220  # random seed to make results reproducible
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    n_classes = n_classes
    # Extract number of chans and time steps from dataset
    n_chans = train_set[0][0].shape[0]
    input_window_samples = train_set[0][0].shape[1]

    model = SampleNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length='auto',
    )

    # Send model to GPU
    if cuda:
        model.cuda()

    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW, #RMSProp
        train_split=predefined_split(valid_set),  # using valid_set for validation
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
    )
    # Model training for a specified number of epochs. `y` is None as it is already supplied
    # in the dataset.
    clf.fit(train_set, y=None, epochs=n_epochs)
    plot_result_sample(clf, dataset_name, train_test_ratio, file_name)











def plot_result_sample(clf, dataset_name, train_test_ratio, file_name):
    '''

    :param clf: (EEGClassifier) from run_sample_model
    :return:
    '''
    # Extract loss and accuracy values for plotting from history object
    results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
    df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                      index=clf.history[:, 'epoch'])

    # get percent of misclass for better visual comparison to loss
    df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
                   valid_misclass=100 - 100 * df.valid_accuracy)

    plt.style.use('seaborn')
    fig, ax1 = plt.subplots(figsize=(8, 3))
    df.loc[:, ['train_loss', 'valid_loss']].plot(
        ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
    ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    df.loc[:, ['train_misclass', 'valid_misclass']].plot(
        ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
    ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
    ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
    ax1.set_xlabel("Epoch", fontsize=14)

    # where some data has already been plotted to ax
    handles = []
    handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
    handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
    plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
    plt.tight_layout()
    plt.title(file_name + '\n'
              "Dataset: " + dataset_name + "    Train/Test Ratio: " + str(train_test_ratio))

    Path("./learning plots").mkdir(parents=True, exist_ok=True) # checks if this directory exists, creates one if not.
    plt.savefig("./learning plots/{}.png".format(file_name), bbox_inches = "tight")    # saves the learning curve plot to the directory