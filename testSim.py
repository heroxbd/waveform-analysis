# -*- coding: utf-8 -*-

import h5py
import numpy as np
# np.seterr(all='raise')
from scipy.stats import uniform

import wf_func as wff

Tau = 40
Sigma = 10
window = 1029

gmu = 160.
p = [8., 0.5, 24.]
p[2] = p[2] * gmu / np.sum(wff.spe(np.arange(window), tau=p[0], sigma=p[1], A=p[2]))
std = 1.

np.random.seed(42)

wdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('Waveform', np.int16, window)])
tdtp = np.dtype([('TriggerNo', np.uint32), ('T0', np.float64)])
sdtp = np.dtype([('TriggerNo', np.uint32), ('PMTId', np.uint32), ('HitPosInWindow', np.float64), ('Charge', np.float64)])
pan = np.arange(window)
npe = np.arange(1, 11, dtype=np.int)
t0 = np.random.uniform(100, 500, size=10)
sams = [np.vstack((wff.time(npe[i], Tau, Sigma) + t0[i], wff.charge(npe[i], gmu=gmu))).T for i in range(10)]
waves = np.empty(10).astype(wdtp)
for i in range(10):
    wave = np.sum([np.where(pan > sams[i][j, 0], wff.spe(pan - sams[i][j, 0], tau=p[0], sigma=p[1], A=p[2]) * sams[i][j, 1] / gmu, 0) for j in range(len(sams[i]))], axis=0)
    wave = wave + np.random.normal(0, std, size=window)
    waves[i]['Waveform'] = np.around(wave).astype(np.int16)
t = np.empty(10).astype(tdtp)
t['TriggerNo'] = np.arange(10).astype(np.uint32)
t['T0'] = t0
waves['TriggerNo'] = np.arange(10).astype(np.uint32)
waves['ChannelID'] = 0
samples = np.empty(sum([len(sams[i]) for i in range(10)])).astype(sdtp)
samples['TriggerNo'] = np.repeat(np.arange(10), [len(sams[i]) for i in range(10)]).astype(np.uint32)
samples['PMTId'] = 0
samples['HitPosInWindow'] = np.hstack([sams[i][:, 0] for i in range(10)])
samples['Charge'] = np.hstack([sams[i][:, 1] for i in range(10)])

assert np.sum(np.logical_not(np.any(np.isnan(waves['Waveform']), axis=1) | np.isin(waves['TriggerNo'], samples['TriggerNo'][samples['HitPosInWindow'] > window - 1]) | np.isin(waves['TriggerNo'], samples['TriggerNo'][samples['HitPosInWindow'] < 0]))) == 10

with h5py.File('mix01.h5', 'w') as opt:
    opt.create_dataset('SimTruth/T', data=t, compression='gzip', compression_opts=4)
    opt.create_dataset('SimTriggerInfo/PEList', data=samples, compression='gzip', compression_opts=4)
    dset = opt.create_dataset('Readout/Waveform', data=waves, compression='gzip', compression_opts=4)
    dset.attrs['mu'] = np.nan
    dset.attrs['tau'] = Tau
    dset.attrs['sigma'] = Sigma
print('mix01.h5 saved, l =', 10)