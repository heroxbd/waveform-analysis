import numpy as np
import numpy.random as random
from scipy import stats
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as Data

from torch import optim
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

import os, sys
import time

import pytorch_stats_loss as stats_loss

BATCHSIZE=16

# Make Saving_Directory
SavePath = "./Network_Models_{}/".format(sys.argv[2])
if not os.path.exists(SavePath):
    os.makedirs(SavePath)

localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
training_record_name = SavePath+"training_record_"+localtime
testing_record_name = SavePath+"testing_record_"+localtime

training_record = open((training_record_name+".txt"),"a+")
testing_record = open((testing_record_name+".txt"),"a+")

# Loading Data
LoadPath= "./Pre-Processing_Results_{}/".format(sys.argv[2])
Data_Name= "ztraining-9_800000-999999"
data_name= LoadPath+Data_Name+".npz"
Data_set= np.load(data_name)
WaveData = Data_set['Wave']
PETData= Data_set['PET']
# Make Shift For +5 ns
PETData = np.concatenate((np.zeros((len(PETData),5)),PETData[:,5:]),axis=-1)
print("Data_loaded")

# Splitting_Data
Wave_train, Wave_test, PET_train, PET_test = train_test_split(WaveData, PETData, test_size=0.2, random_state=42)
print("set_splitted")
print("training_set ",len(Wave_train),", testing_set",len(Wave_test))

# Making Dataset
train_data = Data.TensorDataset(data_tensor=torch.from_numpy(Wave_train).float(),\
                              target_tensor=torch.from_numpy(PET_train).float())
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)

test_data = Data.TensorDataset(data_tensor=torch.from_numpy(Wave_test).float(),\
                              target_tensor=torch.from_numpy(PET_test).float())
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCHSIZE, shuffle=False)

#Neural Networks
class Net_1(nn.Module):

    def __init__(self):
        super(Net_1, self).__init__()

        self.conv1 = nn.Conv1d(1, 25, 9, padding=4)
        self.conv2 = nn.Conv1d(25, 16, 7, padding=3)
        self.conv3 = nn.Conv1d(16, 10, 7, padding=3)
        self.conv4 = nn.Conv1d(10, 1, 1)

    def forward(self, x):
        leaky_relu = nn.LeakyReLU(0.05)
        drop_out = nn.Dropout(0.9)
        x = torch.unsqueeze(x, 1)
        x = leaky_relu(self.conv1(x))
        x = leaky_relu(self.conv2(x))
        x = leaky_relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.squeeze(1)
        return x

net = torch.load("./Network_Models/ztraining-9_600000-799999_epoch2_loss0.8582") # Pre-trained Model Parameters

optimizer = optim.Adam(net.parameters(), lr=1e-5)  # beginning 1e-3
checking_period = np.int(0.25*(len(Wave_train)/BATCHSIZE))

# make loop
training_result = []
testing_result = []
print("training starts")
for epoch in range(7):  # loop over the dataset multiple times # 2 hours for 25 epochs
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = stats_loss.torch_wasserstein_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if (i+1) % checking_period == 0:    # print every 2000 mini-batches
            print('[%d, %5d] running_loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / checking_period))
            training_record.write('%.3f '%((running_loss/checking_period)))
            training_result.append((running_loss/checking_period))
            running_loss = 0.0

    # checking results in testing_s
    if epoch % 1 == 0:     #4 for first 2 rounds, rest 1 for every_round
        batch_result = 0
        batch_count = 0
        for j, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            for batch_index_2 in range(outputs.shape[0]): # range(BATCHSIZE)
                #  the reminder group of BATCHING may not be BATCH_SIZE
                output_vec = outputs.data[batch_index_2].numpy()
                label_vec = labels.data[batch_index_2].numpy()
                if np.sum(label_vec)<=0:
                    label_vec = np.ones(1029)/10000
                    print("warning")
                if np.sum(output_vec)<=0:
                    output_vec = np.ones(1029)/10000
                    print("warning")
                cost = stats.wasserstein_distance(np.arange(1029), np.arange(1029), output_vec, label_vec)
                batch_result += cost
            batch_count += 1
        test_performance = batch_result / (BATCHSIZE * batch_count)
        print("epoch ",str(epoch)," test:",test_performance)
        testing_record.write("%4f " %(test_performance))
        testing_result.append(test_performance)

        # saving network
        save_name = SavePath+Data_Name+"_epoch"+str(epoch)+"_loss"+"%.4f" %(test_performance)
        torch.save(net,save_name)

print('Training Finished')
print(training_result)
print(testing_result)

np.savez(training_record_name,training_result)
np.savez(testing_record_name,testing_result)
training_record.close()
testing_record.close()