import os
import re
import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file prefix')
psr.add_argument('-o', '--output', dest='opt', nargs='+', help='output')
psr.add_argument('--ref', type=str, dest='ref', help='reference file')
psr.add_argument('-n', '--channelid', dest='cid', type=int)
psr.add_argument('-m', '--maxsetnumber', dest='msn', type=int, default=0)
psr.add_argument('-B', '--batchsize', dest='BAT', type=int, default=128)
args = psr.parse_args()

Model = args.opt[0]
SavePath = args.opt[1]
reference = args.ref
ChannelID = args.cid
filename = args.ipt
max_set_number = args.msn
BATCHSIZE = args.BAT

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import torch
torch.manual_seed(42)
import torch.utils.data as Data
from torch import optim
from torch.autograd import Variable
import time
import tables
import wf_func as wff
import loss as stats_loss

# detecting cuda device and wait in line
device = torch.device(ChannelID % 2)
torch.cuda.init()
torch.cuda.empty_cache()
# Make Saving_Directory
if not os.path.exists(SavePath):
    os.makedirs(SavePath)

localtime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
training_record_name = SavePath + 'training_record_' + localtime
testing_record_name = SavePath + 'testing_record_' + localtime

training_record = open((training_record_name + '.txt'), 'a+')
testing_record = open((testing_record_name + '.txt'), 'a+')

# Loading Data
PreFile = tables.open_file(filename, 'r')
if max_set_number > 0 :
    max_set_number = min(max_set_number, len(PreFile.root.Waveform))
else :
    max_set_number = None

print('Reading Data...')
WaveData = PreFile.root.Waveform[0:max_set_number]
PETData = PreFile.root['ChargeSpectrum'][0:max_set_number]
WindowSize = len(WaveData[0])
spe_pre = wff.read_model(reference)
spe = np.concatenate([spe_pre[ChannelID]['spe'], np.zeros(WindowSize - len(spe_pre[ChannelID]['spe']))])
spe = spe / np.sum(spe)
mnecpu = spe[np.mod(np.arange(WindowSize).reshape(WindowSize, 1) - np.arange(WindowSize).reshape(1, WindowSize), WindowSize)]
mne = torch.from_numpy(mnecpu.T).float().to(device=device)
print('Data_loaded')

# Splitting_Data
Wave_train, Wave_test, PET_train, PET_test = train_test_split(WaveData, PETData, test_size=0.05, random_state=42)
print('set_splitted')
print('training_set ', len(Wave_train), ', testing_set', len(Wave_test))

# Making Dataset
train_data = Data.TensorDataset(torch.from_numpy(Wave_train).float().to(device=device),
                                torch.from_numpy(PET_train).float().to(device=device))
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True, pin_memory=False)
test_data = Data.TensorDataset(torch.from_numpy(Wave_test).float().to(device=device),
                               torch.from_numpy(PET_test).float().to(device=device))
# train_data = Data.TensorDataset(torch.from_numpy(Wave_train).float().to(device=device),
#                                 torch.from_numpy(Wave_train).float().to(device=device))
# train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True, pin_memory=False)
# test_data = Data.TensorDataset(torch.from_numpy(Wave_test).float().to(device=device),
#                                torch.from_numpy(Wave_test).float().to(device=device))

test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCHSIZE, shuffle=False, pin_memory=False)

def testing(test_loader, met='wdist') :
    batch_result = 0
    batch_count = 0
    for j, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = net(inputs)
        for batch_index_2 in range(len(outputs)):  # range(BATCHSIZE)
            # the reminder group of BATCHING may not be BATCH_SIZE
            output_vec = outputs.data[batch_index_2].cpu().numpy()
            label_vec = labels.data[batch_index_2].cpu().numpy()
            if np.sum(label_vec) <= 0:
                label_vec = np.ones(WindowSize) / 10000
            if np.sum(output_vec) <= 0:
                output_vec = np.ones(WindowSize) / 10000
            # Wdist loss
            if met == 'wdist':
                cost = stats.wasserstein_distance(np.arange(WindowSize), np.arange(WindowSize), output_vec, label_vec)
            elif met == 'l2':
#                 cost = np.sum(np.power(np.matmul(mnecpu, output_vec) - np.matmul(mnecpu, label_vec), 2)) / (np.sum(output_vec * label_vec > 0) + 1e-3)
                cost = np.linalg.norm(np.matmul(mnecpu, output_vec) - np.matmul(mnecpu, label_vec), ord=2)
#                 cost = np.linalg.norm(np.matmul(mnecpu, output_vec) - label_vec, ord=2)
            batch_result += cost
        batch_count += 1
    return batch_result / (BATCHSIZE * batch_count)

# Neural Networks
from cnnmodule import Net_1

trial_data = Data.TensorDataset(torch.from_numpy(Wave_test[0:1000]).float().to(device=device),
                                torch.from_numpy(PET_test[0:1000]).float().to(device=device))
trial_loader = Data.DataLoader(dataset=trial_data, batch_size=BATCHSIZE, shuffle=False, pin_memory=False)

if os.path.exists(Model) :
    net = torch.load(Model, map_location=device)
    loss = testing(trial_loader)
    lr = 5e-4
else :
    loss = 10000
    while(loss > 100):
        net = Net_1().to(device)
        loss = testing(trial_loader, met='wdist')
        print('Trying initial parameters with loss={:.02f}'.format(loss))
    lr = 5e-3
print('Initial loss={}'.format(loss))
print('Sum of parameters: {:.02f}'.format(sum(parm.numel() for parm in net.parameters())))
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #0.001
optimizer = optim.Adam(net.parameters(), lr=lr)
checking_period = np.int(0.25 * (len(Wave_train) / BATCHSIZE))

# make loop
training_result = []
testing_result = []
print('training start with batchsize={0}'.format(BATCHSIZE))
for epoch in range(49):  # loop over the dataset multiple times
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
#         loss = stats_loss.torch_l2_loss(outputs, labels, mne)
#         loss = stats_loss.torch_l2_loss(outputs, inputs, mne)
        loss = stats_loss.torch_wasserstein_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # running_loss += loss.data[0]
        running_loss += loss.data.item()

        if (i + 1) % checking_period == 0:    # print every 2000 mini-batches
            print('[{0:02d}, {1:05d}] running_loss: {2:.04f}'.format(epoch + 1, i + 1, running_loss / checking_period))
            training_record.write('{:.04f} '.format(running_loss / checking_period))
            training_result.append((running_loss / checking_period))
            running_loss = 0.0

    # checking results in testing_s
    test_performance = testing(test_loader)
#     test_performance = testing(test_loader, met='l2')
    print('epoch ', str(epoch), ' test:', test_performance)
    testing_record.write('{:.04f}'.format(test_performance))
    testing_result.append(test_performance)
    if epoch % 4 == 0:
        # saving network
        save_name = SavePath + '_epoch{0:02d}_loss{1:.04f}'.format(epoch, test_performance)
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
matchrule = re.compile(r'_epoch(\d+)_loss(\d+(\.\d*)?|\.\d+)')
NetLoss_reciprocal = []
for filename in fileSet :
    if '_epoch' in filename : NetLoss_reciprocal.append(1 / float(matchrule.match(filename)[2]))
    else : NetLoss_reciprocal.append(0)
net_name = fileSet[NetLoss_reciprocal.index(max(NetLoss_reciprocal))]
modelpath = '../' + SavePath.split('/')[-2] + '/' + net_name

os.system('ln -snf ' + modelpath + ' ' + Model)