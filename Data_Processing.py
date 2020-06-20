import os
import re
import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file prefix')
psr.add_argument('-o', '--output', dest='opt', help='output')
psr.add_argument('-n', '--channelid', dest='cid', type=int)
psr.add_argument('-m', '--maxsetnumber', dest='msn', type=int, default=0)
psr.add_argument('-B', '--batchsize', dest='BAT', type=int, default=64)
args = psr.parse_args()

SavePath = os.path.dirname(args.opt) + '/'
Model = args.opt
SavePath = args.opt
filename = args.ipt
ChannelID = args.cid
max_set_number = args.msn
BATCHSIZE = args.BAT

import numpy as np
# import numpy.random as random
from scipy import stats
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as Data
from torch import optim
from torch.autograd import Variable

import os
import time
import tables
import pytorch_stats_loss as stats_loss

# detecting cuda device and wait in line
if ChannelID % 2 == 1 :
    device = torch.device(0)
else :
    device = torch.device(1)
# Make Saving_Directory
if not os.path.exists(SavePath):
    os.makedirs(SavePath)

localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
training_record_name = SavePath + "training_record_" + localtime
testing_record_name = SavePath + "testing_record_" + localtime

training_record = open((training_record_name + ".txt"), "a+")
testing_record = open((testing_record_name + ".txt"), "a+")

# Loading Data
PreFile = tables.open_file(filename, 'r')
Data_set = PreFile.root.TrainDataTable
if max_set_number > 0 :
    max_set_number = min(max_set_number, len(Data_set))
else :
    max_set_number = None

print("Reading Data...")
WaveData = (Data_set[0:max_set_number]['Waveform'])
PETData = Data_set[0:max_set_number]['HitSpectrum']
WindowSize = len(WaveData[0])
# Make Shift For +5 ns
PETData = np.concatenate((np.zeros((len(PETData), 5)), PETData[:, 5:]), axis=-1)
print("Data_loaded")

# Splitting_Data
Wave_train, Wave_test, PET_train, PET_test = train_test_split(WaveData, PETData, test_size=0.05, random_state=42)
print("set_splitted")
print("training_set ", len(Wave_train), ", testing_set", len(Wave_test))

# Making Dataset
# train_data = Data.TensorDataset(data_tensor=torch.from_numpy(Wave_train).float(),\
#                              target_tensor=torch.from_numpy(PET_train).float())
train_data = Data.TensorDataset(torch.from_numpy(Wave_train).cuda(device=device).float(),
                                torch.from_numpy(PET_train).cuda(device=device).float())

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True, pin_memory=False)

# test_data = Data.TensorDataset(data_tensor=torch.from_numpy(Wave_test).float(),\
#                              target_tensor=torch.from_numpy(PET_test).float())
test_data = Data.TensorDataset(torch.from_numpy(Wave_test).float().cuda(device=device).float(),
                               torch.from_numpy(PET_test).cuda(device=device).float())

test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCHSIZE, shuffle=False, pin_memory=False)


def testing(test_loader) :
    batch_result = 0
    batch_count = 0
    for j, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = net(inputs)
        for batch_index_2 in range(outputs.shape[0]):  # range(BATCHSIZE)
            #  the reminder group of BATCHING may not be BATCH_SIZE
            output_vec = outputs.data[batch_index_2].cpu().numpy()
            label_vec = labels.data[batch_index_2].cpu().numpy()
            if np.sum(label_vec) <= 0:
                label_vec = np.ones(WindowSize) / 10000
                # print("warning")
            if np.sum(output_vec) <= 0:
                output_vec = np.ones(WindowSize) / 10000
                # print("warning")
            cost = stats.wasserstein_distance(np.arange(WindowSize), np.arange(WindowSize), output_vec, label_vec)
            batch_result += cost
        batch_count += 1
    return batch_result / (BATCHSIZE * batch_count)


# Neural Networks
from CNN_Module import Net_1

trial_data = Data.TensorDataset(torch.from_numpy(Wave_test[0:1000]).cuda(device=device).float(),
                                torch.from_numpy(PET_test[0:1000]).cuda(device=device).float())
trial_loader = Data.DataLoader(dataset=trial_data, batch_size=BATCHSIZE, shuffle=False, pin_memory=False)

if os.path.exists(Model) :
    net = torch.load(Model, map_location=device)
    loss = testing(trial_loader)
    lr = 5e-4
    # BATCHSIZE = BATCHSIZE / 4
else :
    loss = 10000
    while(loss > 100) :
        net = Net_1().to(device)
        loss = testing(trial_loader)
        print("Trying initial parameters with loss={:.2f}".format(loss))
    lr = 1e-2
print("Initial loss={}".format(loss))
print("Sum of parameters: {:.4f}".format(sum(parm.numel() for parm in net.parameters())))
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #0.001
optimizer = optim.Adam(net.parameters(), lr=lr)
checking_period = np.int(0.25 * (len(Wave_train) / BATCHSIZE))

# make loop
training_result = []
testing_result = []
print("training start with batchsize={0}".format(BATCHSIZE))
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

        if (i + 1) % checking_period == 0:    # print every 2000 mini-batches
            print('[%d, %5d] running_loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / checking_period))
            training_record.write('%.3f ' % ((running_loss / checking_period)))
            training_result.append((running_loss / checking_period))
            running_loss = 0.0

    # checking results in testing_s
    if epoch % 4 == 0:
        test_performance = testing(test_loader)
        print("epoch ", str(epoch), " test:", test_performance)
        testing_record.write("%4f " % (test_performance))
        testing_result.append(test_performance)
        # saving network
        save_name = SavePath + "_epoch" + '{:02d}'.format(epoch) + "_loss" + "%.4f" % (test_performance)
        torch.save(net, save_name)

print('Training Finished')
print(training_result)
print(testing_result)

np.savez(training_record_name, training_result)
np.savez(testing_record_name, testing_result)
training_record.close()
testing_record.close()
PreFile.close()

fileSet = os.listdir(SavePath)
matchrule = re.compile(r'(\d+)_epoch(\d+)_loss(\d+(\.\d*)?|\.\d+)')
NetLoss_reciprocal = []
for filename in fileSet :
    if '_epoch' in filename : NetLoss_reciprocal.append(1 / float(matchrule.match(filename)[3]))
    else : NetLoss_reciprocal.append(0)
net_name = fileSet[NetLoss_reciprocal.index(max(NetLoss_reciprocal))]
modelpath = SavePath + net_name

os.system('ln -snf ' + modelpath + ' ' + args.opt)
