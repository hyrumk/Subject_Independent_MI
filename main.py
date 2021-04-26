import torch
from braindecode.util import set_random_seeds
from sampleNet import SampleNet
#from dataprocess import call_data, preprocess_data, create_windows_dataset
from run_model import run_sample_model
from independent_dataprocess import call_data, generate_ss_feature
import spectralNet as sn
import pickle
import numpy as np

#TODO Fix errors that occur in BNCI2014004 data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DATASET_NAME = "BNCI2014001"
    #data_list = call_data(DATASET_NAME, [i for i in range(1,10)])
    #data = call_data(DATASET_NAME, [i for i in range(1,10)])
    '''    #Calls the data and preprocess
    data = call_data(DATASET_NAME, [1,2])
    i = 0
    subject_data = data[i]
    X, Y, frequency_order = generate_ss_feature(data)
    print(frequency_order)
    print(len(X)) # number of trials
    print(len(X[0]))  # should be 20
    print(X[0][0].shape)  # a single spectral input
    print(Y.shape)
    
    
    # for saving in a pkl file
    with open('data_X.pkl','wb') as f:
        pickle.dump(X, f)
    with open('data_Y.pkl', 'wb') as f:
        pickle.dump(Y, f)
    '''
    with open('data_X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('data_Y.pkl', 'rb') as f:
        Y = pickle.load(f)
    Xn = np.array(X)
    print(Y)
    trainloader = sn.numpy_to_trainloader(Xn,Y,100)
    sn.train_and_test(trainloader, 50, 0)

'''
    #run_sample_model(subject_data, dataset_name=DATASET_NAME, file_name="2a subject {} temp".format(str(i + 1)), n_epochs=30)
    run_sample_model(subject_data, dataset_name=DATASET_NAME, train_test_ratio=0.8,
                     file_name="temp", n_epochs=30)   #"2a subject {} shallow no_batchnorm ttratio1".format(str(i + 1))
    run_sample_model(subject_data, dataset_name=DATASET_NAME, train_test_ratio=0.7,
                     file_name="temp", n_epochs=30)   #"2a subject {} shallow no_batchnorm ttratio2".format(str(i + 1))
    run_sample_model(subject_data, dataset_name=DATASET_NAME, train_test_ratio=0.6,
                     file_name="temp", n_epochs=30)   #"2a subject {} shallow no_batchnorm ttratio3".format(str(i + 1))


    for i, subject_data in enumerate(data):
        run_sample_model(subject_data, dataset_name=DATASET_NAME, file_name = "2b subject {} temp".format(str(i+1)), n_epochs=30)
        run_sample_model(subject_data, dataset_name=DATASET_NAME, train_test_ratio = 0.8, file_name="2b subject {} shallow tt ratio change1".format(str(i+1)), n_epochs=30)
        run_sample_model(subject_data, dataset_name=DATASET_NAME, train_test_ratio = 0.7, file_name="2b subject {} shallow tt ratio change2".format(str(i+1)), n_epochs=30)
        run_sample_model(subject_data, dataset_name=DATASET_NAME, train_test_ratio=0.6, file_name="2b subject {} shallow tt ratio change3".format(str(i + 1)), n_epochs=30)
'''

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

