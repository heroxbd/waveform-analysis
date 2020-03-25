from IPython import embed #ipython breakpoint inserting

import numpy as np
import numpy.random as random
from scipy import stats
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as Data

from torch import optim
from torch.autograd import Variable

import os
import sys
import time
import tables

import pytorch_stats_loss as stats_loss

BATCHSIZE=100
BATCHSIZE_fine=10
lr=1e-3
lr_fine=1e-5

fileno=int(sys.argv[-1])

#detecting cuda device and wait in line
if torch.cuda.is_available():
    from Cuda_Queue import *
    while not QueueUp(fileno) : continue # append fileno to waiting list (first line of .bulletin.swp)
    device=wait_in_line(fileno,1024*1024*1024*7.6,0.7)
    torch.cuda.set_device(device)
else : 
    device = 'cpu'
    print('Using device: cpu')
    
# begin loading
# Make Saving_Directory
SavePath = sys.argv[1]
if not os.path.exists(SavePath):
    os.makedirs(SavePath)

localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
training_record_name = SavePath+"training_record_"+localtime
testing_record_name = SavePath+"testing_record_"+localtime

training_record = open((training_record_name+".txt"),"a+")
testing_record = open((testing_record_name+".txt"),"a+")

# Loading Data
LoadPath= sys.argv[2]
'''
prefix = 'ftraining-0'
# Data_Name= "ztraining-9_0-199999"
Data_Name = prefix+'_0-199999'
p 
data_name= LoadPath+Data_Name+".npz"
'''
fileSet = os.listdir(LoadPath)
fileSize = []
for filename in fileSet :
    if '.h5' in filename :
        fileSize.append(os.path.getsize(LoadPath+filename))
    else :
        fileSize.append(0)
data_name = fileSet[fileSize.index(max(fileSize))]
PreFile = tables.open_file(LoadPath+data_name,'r')
Data_set= PreFile.root.TrainDataTable
max_set_number = int(sys.argv[3])
if max_set_number>0 :
    max_set_number = min(max_set_number,len(Data_set))
else :
    max_set_number = -1

print("Reading Data...")
WaveData = Data_set[0:max_set_number]['Wave']
PETData= Data_set[0:max_set_number]['PET']
WindowSize = len(WaveData[0])
#Make Shift For +5 ns
PETData = np.concatenate((np.zeros((len(PETData),5)),PETData[:,5:]),axis=-1)
print("Data_loaded")

# Splitting_Data
Wave_train, Wave_test, PET_train, PET_test = train_test_split(WaveData, PETData, test_size=0.05, random_state=42)
print("set_splitted")
print("training_set ",len(Wave_train),", testing_set",len(Wave_test))

# Making Dataset
#train_data = Data.TensorDataset(data_tensor=torch.from_numpy(Wave_train).float(),\
#                              target_tensor=torch.from_numpy(PET_train).float())
train_data = Data.TensorDataset(torch.from_numpy(Wave_train).float().cuda(device=device),\
                              torch.from_numpy(PET_train).cuda(device=device))

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True, pin_memory=False)

#test_data = Data.TensorDataset(data_tensor=torch.from_numpy(Wave_test).float(),\
#                              target_tensor=torch.from_numpy(PET_test).float())
test_data = Data.TensorDataset(torch.from_numpy(Wave_test).float().cuda(device=device),\
                              torch.from_numpy(PET_test).cuda(device=device))

test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCHSIZE, shuffle=False, pin_memory=False)


# Neural Networks
from CNN_Module import Net_1

net = Net_1().cuda(device=device)

if device!='cpu' :
    ## finish loading to GPU, give tag on .bulletin.swp
    os.system("echo {} {} >> .bulletin.swp".format(fileno,0))

print(sum(parm.numel() for parm in net.parameters()))
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #0.001
optimizer = optim.Adam(net.parameters(), lr=1e-3)
checking_period = np.int(0.25*(len(Wave_train)/BATCHSIZE))

# make loop
training_result = []
testing_result = []
print("training start with batchsize={0}".format(BATCHSIZE))
Fine_Train = True
for epoch in range(25):  # loop over the dataset multiple times
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

        # running_loss += loss.data[0]
        running_loss += loss.data.item()

        if (i+1) % checking_period == 0:    # print every 2000 mini-batches
            print('[%d, %5d] running_loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / checking_period))
            training_record.write('%.3f '%((running_loss/checking_period)))
            training_result.append((running_loss/checking_period))
            running_loss = 0.0

    if epoch > 0 and not Fine_Train :
        if np.mean(training_result[-8:-4])*0.98<np.mean(training_result[-3:]) :
            Fine_Train = True
            BATCHSIZE = 16
            train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE_fine, shuffle=True, pin_memory=False)
            print("Switch to Fine Training with batchsize={0}".format(BATCHSIZE_fine))
                

    # checking results in testing_s
    if epoch % 4 == 0:
        batch_result = 0
        batch_count = 0
        for j, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            for batch_index_2 in range(outputs.shape[0]): # range(BATCHSIZE)
                #  the reminder group of BATCHING may not be BATCH_SIZE
                output_vec = outputs.data[batch_index_2].cpu().numpy()
                label_vec = labels.data[batch_index_2].cpu().numpy()
                if np.sum(label_vec)<=0:
                    label_vec = np.ones(WindowSize)/10000
                    print("warning")
                if np.sum(output_vec)<=0:
                    output_vec = np.ones(WindowSize)/10000
                    print("warning")
                cost = stats.wasserstein_distance(np.arange(WindowSize), np.arange(WindowSize), output_vec, label_vec)
                batch_result += cost
            batch_count += 1
        test_performance = batch_result / (BATCHSIZE * batch_count)
        print("epoch ",str(epoch)," test:",test_performance)
        testing_record.write("%4f " %(test_performance))
        testing_result.append(test_performance)

        # saving network
        save_name = SavePath+"_epoch"+str(epoch)+"_loss"+"%.4f" %(test_performance)
        torch.save(net,save_name)

print('Training Finished')
print(training_result)
print(testing_result)

np.savez(training_record_name,training_result)
np.savez(testing_record_name,testing_result)
training_record.close()
testing_record.close()
PreFile.close()
