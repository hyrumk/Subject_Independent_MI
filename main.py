import torch
from braindecode.util import set_random_seeds
from sampleNet import SampleNet
#from dataprocess import call_data, preprocess_data, create_windows_dataset
from run_model import run_sample_model, run_shallow_net
from independent_dataprocess import call_data, generate_ss_feature, generate_ss_feature_test, concat_dataset, bandpass_window_BaseConcat
import spectralNet as sn
import pickle
import numpy as np
import argparse
from distutils.util import strtobool
import os


def main(args):
    DATASET_NAME = "BNCI2014001"
    data_list = call_data(DATASET_NAME, [i for i in range(1,10)])

    subject_list = args.import_data if len(args.import_data) else [i for i in range(9)]  # 9 is for 2a specific. <TODO> Needs a change when expanding to another dataset
    input_exists = True
    for i in subject_list:
        subject = i+1
        if os.path.exists('data/data_train_X{}.pkl'.format(subject)) and os.path.exists('data/data_train_Y{}.pkl'.format(subject)) and\
            os.path.exists('data/data_freq_order{}.pkl'.format(subject)) and os.path.exists('data/data_test_X{}.pkl'.format(subject)) and\
            os.path.exists('data/data_test_Y{}.pkl'.format(subject)):
            # file exists
            pass
        else:
            input_exists = False
            break

    # required input,labels not present (spatial-spectral data, both training and test data according to a target subject)
    if not input_exists:
        for subject in subject_list:
            # IMPORTING bcicompetition data via braindecode
            data = []   # a list of all training subjects (list[MOABB dataset])
            data_test = []  # the target test subject (list[MOABB dataset])
            CHOSEN_TEST_SUBJECT = subject + 1

            for i in range(len(subject_list)):
                if i == CHOSEN_TEST_SUBJECT - 1:
                    subject_data = data_list[i]
                    data_test.append(subject_data)
                    continue
                subject_data = data_list[i]
                data.append(subject_data)


            # Where you use the braindecode_data to make a filtered data
            X, Y, frequency_order = generate_ss_feature(data) # train data
            X_test, Y_test = generate_ss_feature_test(data_test, frequency_order[:20])  # test data

            # save generated train data
            with open('data/data_train_X{}.pkl'.format(CHOSEN_TEST_SUBJECT), 'wb') as f:
                pickle.dump(X, f)
            with open('data/data_train_Y{}.pkl'.format(CHOSEN_TEST_SUBJECT), 'wb') as f:
                pickle.dump(Y, f)
            with open('data/data_freq_order{}.pkl'.format(CHOSEN_TEST_SUBJECT), 'wb') as f:
                pickle.dump(frequency_order, f)

            # save generated test data
            with open('data/data_test_X{}.pkl'.format(CHOSEN_TEST_SUBJECT),'wb') as f:
                pickle.dump(X_test, f)
            with open('data/data_test_Y{}.pkl'.format(CHOSEN_TEST_SUBJECT), 'wb') as f:
                pickle.dump(Y_test, f)



    # Running spectral-spatial net
    final_result_string = ""
    for subject in subject_list:
        CHOSEN_TEST_SUBJECT = subject + 1
        with open('data/data_train_X{}_binary.pkl'.format(CHOSEN_TEST_SUBJECT), 'rb') as f:
            X = pickle.load(f)
        with open('data/data_train_Y{}_binary.pkl'.format(CHOSEN_TEST_SUBJECT), 'rb') as f:
            Y = pickle.load(f)
        with open('data/data_freq_order{}_binary.pkl'.format(CHOSEN_TEST_SUBJECT), 'rb') as f:
            frequency_order = pickle.load(f)


        with open('data/data_test_X{}_binary.pkl'.format(CHOSEN_TEST_SUBJECT),'rb') as f:
            X_test = pickle.load(f)
        with open('data/data_test_Y{}_binary.pkl'.format(CHOSEN_TEST_SUBJECT),'rb') as f:
            Y_test = pickle.load(f)


        X_train = np.array(X)

        batch_size = args.batch_size
        epoch = args.epoch

        trainloader = sn.numpy_to_trainloader(X_train,Y,batch_size)
        testloader = sn.numpy_to_trainloader(X_test, Y_test, batch_size)
        result_string = sn.train_and_test(trainloader, epoch, testloader, num_classes=2)
        print("FINISHED WITH SUBJECT {} TEST".format(CHOSEN_TEST_SUBJECT))
        final_result_string += "SUBJECT {} TEST\n".format(CHOSEN_TEST_SUBJECT)
        final_result_string += result_string

    print("\n\n\n", final_result_string)



# <TODO> Give a range of selection in BCI dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_input", type=strtobool, default="true") # loads the generated data if you have it already
    parser.add_argument("--import_data", type=list, default=[])   #imports braindecode data of selection and preprocess.
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=150)

    args = parser.parse_args()

    main(args)

'''
download
import
run model
epoch
test_subject


not yet: save (model)
 

'''



#TODO Fix errors that occur in BNCI2014004 data

# Press the green button in the gutter to run the script.

#DATASET_NAME = "BNCI2014001"
#data_list = call_data(DATASET_NAME, [i for i in range(1,10)])
#data = call_data(DATASET_NAME, [i for i in range(1,10)])
'''
# import BCI competition 2a data
for i in range(9):
    if i == 0: continue
    data = []
    data_test = []
    CHOSEN_TEST_SUBJECT = i+1

    for i in range(9):
        if i == CHOSEN_TEST_SUBJECT - 1:
            with open('data/braindecode2a_data{}.pkl'.format(i), 'rb') as f:
                subject_data = pickle.load(f)
                data_test.append(subject_data)
            continue
        with open('data/braindecode2a_data{}.pkl'.format(i), 'rb') as f:
            subject_data = pickle.load(f)
            data.append(subject_data)

    # Where you use the data to make a filtered data
    #X, Y, frequency_order = generate_ss_feature(data, binary = True)  # generate train data

    #X_test, Y_test = generate_ss_feature_test(data_test, frequency_order[:20], binary = True)  # generate test data
    X_train, Y_train = concat_dataset(data)
    test_data = bandpass_window_BaseConcat(data_test[0])
    run_shallow_net(X_train, Y_train, test_data)
'''


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

