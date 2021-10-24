import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file')
psr.add_argument('-o', dest='opt', help='output')
psr.add_argument('-N', dest='NetDir', help='Network directory')
psr.add_argument('--met', type=str, help='method')
psr.add_argument('--ref', type=str, nargs='+', help='reference file')
psr.add_argument('-B', '--batchsize', dest='BAT', type=int, default=5)
psr.add_argument('-D', '--device', dest='Device', type=str, default='cpu')
args = psr.parse_args()
NetDir = args.NetDir
filename = args.ipt
output = args.opt
BATCHSIZE = args.BAT
Device = args.Device
reference = args.ref
method = args.met

import time
global_start = time.time()
cpu_global_start = time.process_time()
import numpy as np
from scipy import optimize as opti
import tables
import pandas as pd
from tqdm import tqdm
import h5py

import torch
from torch.nn import functional as F

from multiprocessing import Pool, cpu_count
import wf_func as wff

def Read_Data(startentry, endentry):
    RawDataFile = tables.open_file(filename, 'r')
    WaveformTable = RawDataFile.root.Readout.Waveform
    Waveforms_and_info = WaveformTable[startentry:endentry]
    Shifted_Waves_and_info = np.empty(Waveforms_and_info.shape, dtype=gpufloat_dtype)
    for name in origin_dtype.names:
        if name != 'Waveform':
            Shifted_Waves_and_info[name] = Waveforms_and_info[name]
    for i in range(len(Waveforms_and_info)):
        channelid = Waveforms_and_info[i]['ChannelID']
        Shifted_Waves_and_info[i]['Waveform'] = Waveforms_and_info[i]['Waveform'].astype(np.float64) * spe_pre[channelid]['epulse']
    RawDataFile.close()
    return pd.DataFrame({name: list(Shifted_Waves_and_info[name]) for name in gpufloat_dtype.names})

# Loading Data
RawDataFile = tables.open_file(filename, 'r')
origin_dtype = RawDataFile.root.Readout.Waveform.dtype
Total_entries = len(RawDataFile.root.Readout.Waveform)
RawDataFile.close()
WindowSize = wff.window
gpufloat_dtype = np.dtype([(name, np.dtype('float32') if name == 'Waveform' else origin_dtype[name].base, origin_dtype[name].shape) for name in origin_dtype.names])
print('Initialization finished, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - global_start, time.process_time() - cpu_global_start))
print('Processing {} entries'.format(Total_entries))

with h5py.File(filename, 'r', libver='latest', swmr=True) as ipt:
    l = len(ipt['Readout/Waveform'])
    Mu = ipt['Readout/Waveform'].attrs['mu']
    Tau = ipt['Readout/Waveform'].attrs['tau']
    Sigma = ipt['Readout/Waveform'].attrs['sigma']

N = 10
tic = time.time()
cpu_tic = time.process_time()
spe_pre = wff.read_model(reference[0])
slices = np.append(np.arange(0, Total_entries, int(np.ceil(Total_entries / N))), Total_entries)
ranges = list(zip(slices[0:-1], slices[1:]))
with Pool(min(N, cpu_count())) as pool :
    Waveforms_and_info = pd.concat(pool.starmap(Read_Data, ranges))
print('Data Loaded, consuming {0:.02f}s using {1} threads, cpu time {2:.02f}s'.format(time.time() - tic, N, time.process_time() - cpu_tic))

channelid_set = set(Waveforms_and_info['ChannelID'])
Channel_Grouped_Waveform = Waveforms_and_info.groupby(by='ChannelID')

# Loading CNN Net
tic = time.time()
device_gpu = torch.device(int(Device) if Device != 'cpu' else Device)
device_cpu = torch.device('cpu')
nets_gpu = dict([])
for channelid in tqdm(channelid_set, desc='Loading Nets of each channel') :
    nets_gpu[channelid] = torch.load(NetDir + '/Channel{:02d}.torch_net'.format(channelid), map_location=device_gpu)
nets_cpu = dict([])
for channelid in tqdm(channelid_set, desc='Loading Nets of each channel') :
    nets_cpu[channelid] = torch.load(NetDir + '/Channel{:02d}.torch_net'.format(channelid), map_location=device_cpu)
print('Net Loaded, consuming {0:.02f}s'.format(time.time() - tic))

filter_limit = 0.05
Timeline = np.arange(WindowSize).reshape(1, WindowSize)

p = spe_pre[0]['parameters']
t_auto = np.arange(WindowSize).reshape(WindowSize, 1) - np.arange(WindowSize).reshape(1, WindowSize)
mnecpu = wff.spe((t_auto + np.abs(t_auto)) / 2, p[0], p[1], p[2])

def Forward(channelid, device, nets):
    Data_of_this_channel = Channel_Grouped_Waveform.get_group(channelid)
    Shifted_Wave = np.vstack(Data_of_this_channel['Waveform'])
    TriggerNos = np.array(Data_of_this_channel['TriggerNo'])
    ChannelIDs = np.array(Data_of_this_channel['ChannelID'])
    HitPosInWindows = np.empty(0, dtype=np.int16)
    PEmeasure = np.empty(0, dtype=np.float32)
    EventData = np.empty(0, dtype=np.int64)
    slices = np.append(np.arange(0, len(Shifted_Wave), BATCHSIZE), len(Shifted_Wave))
    time_cnn = np.empty(len(Shifted_Wave))
    for i in range(len(slices) - 1):
        time_cnn_start = time.time()
        inputs = Shifted_Wave[slices[i]:slices[i + 1]]
        Prediction = nets[channelid].forward(torch.from_numpy(inputs).to(device=device)).data.cpu().numpy()
        # Total = np.clip(np.abs(inputs.sum(axis=1)) / wff.gmu, 1e-6 / wff.gmu, np.inf)
        Total = Prediction.sum(axis=1)

        Alpha = np.empty(slices[i + 1] - slices[i])
        for j in range(slices[i + 1] - slices[i]):
            Alpha[j] = opti.fmin_l_bfgs_b(lambda alpha: wff.rss_alpha(alpha, Prediction[j], inputs[j], mnecpu), x0=[0.01], approx_grad=True, bounds=[[1e-20, np.inf]], maxfun=50000)[0]
        Total = Total * Alpha

        sumPrediction = np.clip(Prediction.sum(axis=1), 1e-10, np.inf)
        Prediction = Prediction / sumPrediction[:, None] * Total[:, None]
        HitPosInWindow = Prediction > filter_limit
        pe_numbers = HitPosInWindow.sum(axis=1)
        no_pe_found = pe_numbers == 0
        if no_pe_found.any():
            guessed_risetime = np.around(inputs[no_pe_found].argmax(axis=1) - spe_pre[channelid]['peak_c'])
            guessed_risetime = np.where(guessed_risetime > 0, guessed_risetime, 0)
            HitPosInWindow[no_pe_found, guessed_risetime] = True
            Prediction[no_pe_found, guessed_risetime] = 1
            pe_numbers[no_pe_found] = 1
        sumPrediction = Prediction.sum(axis=1)
        Prediction = Prediction / sumPrediction[:, None] * Total[:, None]
        Prediction = Prediction[HitPosInWindow] * wff.gmu
        PEmeasure = np.append(PEmeasure, Prediction)
        TimeMatrix = np.repeat(Timeline, len(HitPosInWindow), axis=0)[HitPosInWindow]
        HitPosInWindows = np.append(HitPosInWindows, TimeMatrix)
        EventData = np.append(EventData, np.repeat(TriggerNos[slices[i]:slices[i + 1]], pe_numbers))
        time_cnn[slices[i]:slices[i + 1]] = (time.time() - time_cnn_start) / (slices[i + 1] - slices[i])
    ChannelData = np.empty(EventData.shape, dtype=np.int16)
    ChannelData.fill(channelid)
    return pd.DataFrame({'HitPosInWindow': HitPosInWindows, 'Charge': PEmeasure, 'TriggerNo': EventData, 'ChannelID': ChannelData}), time_cnn, TriggerNos, ChannelIDs

tidt = np.dtype([('consumption', np.float64)])
time_cnn_gpu = np.empty(0)
time_cnn_cpu = np.empty(0)
cpu_tic = time.process_time()
Result = []
r_for_sort = np.empty(l, dtype=np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32)]))
for ch in tqdm(channelid_set, desc='Predict for each channel'):
    result_i, time_cnn_gpu_i, TriggerNo_i, ChannelID_i = Forward(ch, device_gpu, nets_gpu)
    Result.append(result_i)
    r_for_sort['TriggerNo'][len(time_cnn_gpu):len(time_cnn_gpu)+len(TriggerNo_i)] = TriggerNo_i
    r_for_sort['ChannelID'][len(time_cnn_gpu):len(time_cnn_gpu)+len(ChannelID_i)] = ChannelID_i
    time_cnn_gpu = np.append(time_cnn_gpu, time_cnn_gpu_i)
    _, time_cnn_cpu_i, _, _ = Forward(ch, device_cpu, nets_cpu)
    time_cnn_cpu = np.append(time_cnn_cpu, time_cnn_cpu_i)
Result = pd.concat(Result)
time_cnn_gpu = time_cnn_gpu[np.argsort(r_for_sort, order=['TriggerNo', 'ChannelID'])]
time_cnn_cpu = time_cnn_cpu[np.argsort(r_for_sort, order=['TriggerNo', 'ChannelID'])]
Result = Result.sort_values(by=['TriggerNo', 'ChannelID'])
Result = Result.to_records(index=False)
ts_gpu = np.zeros(l, dtype=tidt)
ts_gpu['consumption'] = time_cnn_gpu
print('Prediction generated, real time {0:.02f}s, cpu time {1:.02f}s'.format(ts_gpu['consumption'].sum(), time.process_time() - cpu_tic))
ts_cpu = np.zeros(l, dtype=tidt)
ts_cpu['consumption'] = time_cnn_cpu

with h5py.File(output, 'w') as opt:
    dset = opt.create_dataset('photoelectron', data=Result, compression='gzip')
    dset.attrs['Method'] = method
    dset.attrs['mu'] = Mu
    dset.attrs['tau'] = Tau
    dset.attrs['sigma'] = Sigma
    tsdset = opt.create_dataset('starttime', data=ts_gpu, compression='gzip')
    ts_cpu_dset = opt.create_dataset('starttime_cpu', data=ts_cpu, compression='gzip')
    print('The output file path is {}'.format(output))

print('Finished! Consuming {0:.02f}s in total, cpu time {1:.02f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))
