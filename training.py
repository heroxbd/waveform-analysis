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
alpha_Model = args.opt[1]
SavePath = args.opt[2]
reference = args.ref
ChannelID = args.cid
filename = args.ipt
max_set_number = args.msn
BATCHSIZE = args.BAT

import time

import torch
torch.manual_seed(42)
import torch.utils.data as Data
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from scipy import stats
from scipy import optimize as opti
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
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
cnn_training_record_name = SavePath + 'cnn_training_record_' + localtime
cnn_testing_record_name = SavePath + 'cnn_testing_record_' + localtime
alpha_training_record_name = SavePath + 'alpha_training_record_' + localtime
alpha_testing_record_name = SavePath + 'alpha_testing_record_' + localtime

cnn_training_record = open((cnn_training_record_name + '.txt'), 'a+')
cnn_testing_record = open((cnn_testing_record_name + '.txt'), 'a+')
alpha_training_record = open((alpha_training_record_name + '.txt'), 'a+')
alpha_testing_record = open((alpha_testing_record_name + '.txt'), 'a+')

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
p = spe_pre[ChannelID]['parameters']
t_auto = np.arange(WindowSize).reshape(WindowSize, 1) - np.arange(WindowSize).reshape(1, WindowSize)
mnecpu = wff.spe((t_auto + np.abs(t_auto)) / 2, p[0], p[1], p[2])
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
            # RSS loss
            elif met == 'l2':
                # cost = np.sum(np.power(np.matmul(mnecpu, output_vec) - np.matmul(mnecpu, label_vec), 2)) / (np.sum(output_vec * label_vec > 0) + 1e-3)
                cost = np.linalg.norm(np.matmul(mnecpu, output_vec) - np.matmul(mnecpu, label_vec), ord=2)
                # cost = np.linalg.norm(np.matmul(mnecpu, output_vec) - label_vec, ord=2)
            batch_result += cost
        batch_count += 1
    return batch_result / (BATCHSIZE * batch_count)

def alpha_testing(test_loader) :
    batch_result = 0
    batch_count = 0
    for j, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = net(inputs)
        outputs_alpha = alpha(outputs.detach())
        for batch_index_2 in range(len(outputs_alpha)):  # range(BATCHSIZE)
            # the reminder group of BATCHING may not be BATCH_SIZE
            output_vec = outputs_alpha.data[batch_index_2].cpu().numpy()
            input_vec = inputs.data[batch_index_2].cpu().numpy()
            # RSS loss
            cost = np.linalg.norm(np.matmul(mnecpu, output_vec) - input_vec, ord=2)
            batch_result += cost
        batch_count += 1
    return batch_result / (BATCHSIZE * batch_count)

# Neural Networks
from cnnmodule import Net
from cnnmodule import Alpha

trial_data = Data.TensorDataset(torch.from_numpy(Wave_test[0:1000]).float().to(device=device),
                                torch.from_numpy(PET_test[0:1000]).float().to(device=device))
trial_loader = Data.DataLoader(dataset=trial_data, batch_size=BATCHSIZE, shuffle=False, pin_memory=False)

if os.path.exists(Model) :
    net = torch.load(Model, map_location=device)
    alpha = torch.load(alpha_Model, map_location=device)
    loss = testing(trial_loader)
    lr = 5e-4
else :
    loss = 10000
    while(loss > 100):
        net = Net().to(device)
        alpha = Alpha().to(device)
        loss = testing(trial_loader, met='wdist')
        print('Trying initial parameters with loss={:.02f}'.format(loss))
    lr = 5e-3
print('Initial loss={}'.format(loss))
print('Total number of parameters: {:d}'.format(sum(parm.numel() for parm in net.parameters())))
print('First Alpha: {:.06f}'.format(F.softplus(list(alpha.parameters())[0]).data.item()))
# cnn_optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #0.001
cnn_optimizer = optim.Adam(net.parameters(), lr=lr)
alpha_optimizer = optim.Adam(alpha.parameters(), lr=1e-6)
checking_period = np.int(0.25 * (len(Wave_train) / BATCHSIZE))

# make loop
cnn_training_result = []
cnn_testing_result = []
print('training start with batchsize={0}'.format(BATCHSIZE))
Epoch = 1 * 4 + 1
for epoch in range(Epoch):  # loop over the dataset multiple times
    cnn_running_loss = 0.0
    alpha_running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        cnn_optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = stats_loss.torch_wasserstein_loss(outputs, labels)
        loss.backward()
        cnn_optimizer.step()

        # cnn_running_loss += loss.data[0]
        cnn_running_loss += loss.data.item()

        if (i + 1) % checking_period == 0:    # print every 2000 mini-batches
            print('[{0:02d}, {1:05d}] cnn_loss: {2:.04f}'.format(epoch + 1, i + 1, cnn_running_loss / checking_period))
            cnn_training_record.write('{:.04f} '.format(cnn_running_loss / checking_period))
            cnn_training_result.append((cnn_running_loss / checking_period))
            cnn_running_loss = 0.0

    # checking results in testing_s
    cnn_test_performance = testing(test_loader)
    # cnn_test_performance = testing(test_loader, met='l2')
    print('epoch: {0:02d} test: {1:0.4f}'.format(epoch, cnn_test_performance))
    cnn_testing_record.write('{:.04f}'.format(cnn_test_performance))
    cnn_testing_result.append(cnn_test_performance)
    if epoch % 4 == 0:
        # saving network
        save_name = SavePath + '_cnn_epoch{0:02d}_loss{1:.04f}'.format(epoch, cnn_test_performance)
        torch.save(net, save_name)

alpha_training_result = []
alpha_testing_result = []
Epoch = 1 * 4 + 1
for epoch in range(Epoch):  # loop over the dataset multiple times
    alpha_running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        alpha_optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        outputs_alpha = alpha(outputs.detach())
        loss_alpha = stats_loss.torch_l2_loss(outputs_alpha, inputs, mne)
        loss_alpha.backward()
        alpha_optimizer.step()

        alpha_running_loss += loss_alpha.data.item()

        if (i + 1) % checking_period == 0:    # print every 2000 mini-batches
            print('[{0:02d}, {1:05d}] alpha_loss: {2:.04f}'.format(epoch + 1, i + 1, alpha_running_loss / checking_period))
            a = F.softplus(list(alpha.parameters())[0]).data.item()
            print('alpha: {:.02e}'.format(a))
            alpha_training_record.write('{:.04f},{:.02e}\n'.format(alpha_running_loss / checking_period, a))
            alpha_training_result.append((alpha_running_loss / checking_period))
            alpha_running_loss = 0.0

    # checking results in testing_s
    alpha_test_performance = alpha_testing(test_loader)
    print('epoch: {0:02d} test: {1:0.4f}'.format(epoch, alpha_test_performance))
    alpha_testing_record.write('{:.04f}'.format(alpha_test_performance))
    alpha_testing_result.append(alpha_test_performance)
    if epoch % 4 == 0:
        # saving network
        save_name = SavePath + '_alpha_epoch{0:02d}_loss{1:.04f}'.format(epoch, alpha_test_performance)
        torch.save(alpha, save_name)

def rss_alpha(alpha, outputs, inputs, mnecpu):
    r = np.power(alpha * outputs @ mnecpu - inputs, 2).sum()
    return r

slices = np.append(np.arange(0, len(Wave_test), BATCHSIZE), len(PET_test))
alpha_array = np.empty(len(Wave_test))
for i in tqdm(range(len(slices) - 1)):
    a = slices[i]
    b = slices[i + 1]
    inputs = Wave_test[a:b]
    outputs = net.forward(torch.from_numpy(inputs).to(device=device)).data.cpu().numpy()
    for j in range(a, b):
        alpha_array[j] = opti.fmin_l_bfgs_b(lambda alpha: rss_alpha(alpha, outputs[j - a], inputs[j - a], mnecpu), x0=[1], approx_grad=True, bounds=[[0, np.inf]], maxfun=50000)[0]

print('alpha mean = {:.02e}, std = {:.02e}'.format(alpha_array.mean(), alpha_array.std(ddof=-1)))
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.hist(alpha_array, bins=100)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('Count')
ax.grid()
ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
fig.savefig(SavePath + 'alpha.png')
plt.close()

print('Training Finished')
print(cnn_training_result)
print(cnn_testing_result)
print(alpha_training_result)
print(alpha_testing_result)

np.savez(cnn_training_record_name, cnn_training_result)
np.savez(cnn_testing_record_name, cnn_testing_result)
np.savez(alpha_training_record_name, alpha_training_result)
np.savez(alpha_testing_record_name, alpha_testing_result)

cnn_training_record.close()
cnn_testing_record.close()
alpha_training_record.close()
alpha_testing_record.close()
PreFile.close()

fileSet = os.listdir(SavePath)
matchrule = re.compile(r'_cnn_epoch(\d+)_loss(\d+(\.\d*)?|\.\d+)')
NetLoss_reciprocal = []
for filename in fileSet :
    if '_cnn' in filename : NetLoss_reciprocal.append(1 / float(matchrule.match(filename)[2]))
    else : NetLoss_reciprocal.append(0)
net_name = fileSet[NetLoss_reciprocal.index(max(NetLoss_reciprocal))]
modelpath = '../' + SavePath.split('/')[-2] + '/' + net_name

fileSet = os.listdir(SavePath)
matchrule = re.compile(r'_alpha_epoch(\d+)_loss(\d+(\.\d*)?|\.\d+)')
NetLoss_reciprocal = []
for filename in fileSet :
    if '_alpha' in filename : NetLoss_reciprocal.append(1 / float(matchrule.match(filename)[2]))
    else : NetLoss_reciprocal.append(0)
alpha_net_name = fileSet[NetLoss_reciprocal.index(max(NetLoss_reciprocal))]
modelpath = '../' + SavePath.split('/')[-2] + '/' + net_name
alpha_modelpath = '../' + SavePath.split('/')[-2] + '/' + alpha_net_name

os.system('ln -snf ' + modelpath + ' ' + Model)
os.system('ln -snf ' + alpha_modelpath + ' ' + alpha_Model)