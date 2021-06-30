# Subject_Independent_MI

#### A rough implementation of the paper 

#### O. -Y. Kwon, M. -H. Lee, C. Guan and S. -W. Lee, "Subject-Independent Brain–Computer Interfaces Based on Deep Convolutional Neural Networks," in IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 10, pp. 3839-3852, Oct. 2020, doi: 10.1109/TNNLS.2019.2946869.


This implementation only takes BCI4dataset_2a_BNCI2014001 imported using braindecode only at this point.
The model requires you to generate the spectral-spatial input with the EEG dataset you have to use it as a model input.

Command
```
python main.py --download_input true --import_data 1,2,3 --refresh_feature false --batch_size 100 --epoch 150
```
will download the spectral-spatial input from a selected subjects which are subject 1,2,and 3. 
The batch size of the data will be 100 and will run the deep convolutional neural network for 150 epochs in training.

```--download_input ``` will receive either ```true``` or ```false``` as an argument and its default value is ```false```. When it's set to ```true```, it will download "all" the generated spectral-spatial input and the frequency order which are necessary in training and testing the model.

```--import_data``` receives subject numbers, each number separated by a comma without any whitespace. By default, it will import dataset from all 9 subjects. e.g. --import_data 1,2,3 will use the dataset of 3 subjects and test the model on subject 1,2, and 3. 

```--refresh_feature``` decides whether you want to update your spectral-spatial input based on the imported data. For instance, if you have a test dataset of subject 5 created based on subject 1,2,3,4, but you imported data 1,2,and 5 this time and want to recreate a test dataset of subject 5 based on subject 1 and 2 instead of 1,2,3, and 4, setting ```refresh_feature``` as ```true``` will update the test dataset accordingly.

```--batch_size``` sets the batch size of the dataset.

```--epoch``` sets the number of epoch you want to train using the model.


##### *NOTE: This is an implementation done on a personal level. The code is very unorganized at the moment and the input it can take is very limited, but I'm planning on updating the code consistently.

for some reason, create_windows_from_events function used in bandpass_data in independent_dataprocess.py comes up with the list index out of range error, which does not occur when run on colab. 

