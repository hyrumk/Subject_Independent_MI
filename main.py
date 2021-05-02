import torch
from braindecode.util import set_random_seeds
from sampleNet import SampleNet
#from dataprocess import call_data, preprocess_data, create_windows_dataset
from run_model import run_sample_model
from independent_dataprocess import call_data, generate_ss_feature, generate_ss_feature_test
import spectralNet as sn
import pickle
import numpy as np

#TODO Fix errors that occur in BNCI2014004 data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DATASET_NAME = "BNCI2014001"
    #data_list = call_data(DATASET_NAME, [i for i in range(1,10)])
    #data = call_data(DATASET_NAME, [i for i in range(1,10)])

    data = []
    data_test = []
    CHOSEN_TEST_SUBJECT = 9

    for i in range(9):
        if i == CHOSEN_TEST_SUBJECT-1:
            with open('data/braindecode2a_data{}.pkl'.format(i), 'rb') as f:
                subject_data = pickle.load(f)
                data_test.append(subject_data)
            continue
        with open('data/braindecode2a_data{}.pkl'.format(i), 'rb') as f:
            subject_data = pickle.load(f)
            data.append(subject_data)


    #Where you use the data to make a filtered data
    #X, Y, frequency_order = generate_ss_feature(data) # train data

    with open('data_train_X{}.pkl'.format(CHOSEN_TEST_SUBJECT),'rb') as f:
        X = pickle.load(f)
    with open('data_train_Y{}.pkl'.format(CHOSEN_TEST_SUBJECT),'rb') as f:
        Y = pickle.load(f)
    with open('data_freq_order{}.pkl'.format(CHOSEN_TEST_SUBJECT),'rb') as f:
        frequency_order = pickle.load(f)


    X_test, Y_test = generate_ss_feature_test(data_test, frequency_order[:20])  # test data
    with open('data_test_X{}.pkl'.format(CHOSEN_TEST_SUBJECT),'wb') as f:
        pickle.dump(X_test, f)
    with open('data_test_Y{}.pkl'.format(CHOSEN_TEST_SUBJECT), 'wb') as f:
        pickle.dump(Y_test, f)

    with open('data_test_X{}.pkl'.format(CHOSEN_TEST_SUBJECT),'rb') as f:
        X_test = pickle.load(f)
    with open('data_test_Y{}.pkl'.format(CHOSEN_TEST_SUBJECT),'rb') as f:
        Y_test = pickle.load(f)


    # X_train, Y_train, frequency_order = generate_ss_feature(data[:-1])
    # X_test, Y_test = generate_ss_feature_test(data[-1], frequency_order)
    X_train = np.array(X)


    #X_test = np.array(X_test)
    trainloader = sn.numpy_to_trainloader(X_train,Y,100)
    testloader = sn.numpy_to_trainloader(X_test, Y_test, 100)
    sn.train_and_test(trainloader, 200, testloader)

'''

    This brings pickled data (from generate_ss_feature)
    with open('data_X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('data_Y.pkl', 'rb') as f:
        Y = pickle.load(f)
    
    
    #### FOR LOADING preprocessed TRAINING DATA#####  
    with open('data_train_X{}.pkl'.format(CHOSEN_TEST_SUBJECT),'rb') as f:
        X = pickle.load(f)
    with open('data_train_Y{}.pkl'.format(CHOSEN_TEST_SUBJECT),'rb') as f:
        Y = pickle.load(f)
    with open('data_freq_order{}.pkl'.format(CHOSEN_TEST_SUBJECT),'rb') as f:
        frequency_order = pickle.load(f)
    ###################################################

####################################
    # Saving part
    with open('data_train_X{}.pkl'.format(CHOSEN_TEST_SUBJECT),'wb') as f:
        pickle.dump(X, f)
    with open('data_train_Y{}.pkl'.format(CHOSEN_TEST_SUBJECT), 'wb') as f:
        pickle.dump(Y, f)
    with open('data_freq_order{}.pkl'.format(CHOSEN_TEST_SUBJECT),'wb') as f:
        pickle.dump(frequency_order, f)
##############################################
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

