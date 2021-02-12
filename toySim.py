import os
import itertools
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

import h5py
from tqdm import tqdm
from tqdm import trange
import numpy as np
# np.seterr(all='raise')
import scipy.optimize as optimize
import scipy.interpolate as interpolate
from scipy.stats import poisson, uniform, norm, chi2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('--Ncpu', dest='Ncpu', type=int, default=50)
psr.add_argument('-N', dest='N', type=int, default=1e5)
psr.add_argument('--mts', dest='mts', type=str, help='mu & tau & sigma')
psr.add_argument('--noi', dest='noi', action='store_true', help='noise bool', default=False)
args = psr.parse_args()

Ncpu = 100
window = 1029
mtslist = args.mts.split('-')
Mu = float(mtslist[0])
Tau = float(mtslist[1])
Sigma = float(mtslist[2])

gmu = 160.
gsigma = 40.
p = [8., 0.5, 24.]
p[2] = p[2] * gmu / np.sum(wff.spe(np.arange(window), tau=p[0], sigma=p[1], A=p[2]))
std = 1.

def sampling(a0, a1, mu, tau, sigma):
    np.random.seed(a0)
    npe = poisson.ppf(1 - uniform.rvs(scale=1-poisson.cdf(0, mu), size=a1 - a0), mu).astype(np.int)
    t0 = np.random.uniform(100, 500, size=a1 - a0)
    sams = [np.vstack((wff.time(npe[i], tau, sigma) + t0[i], wff.charge(npe[i], gmu=gmu, gsigma=gsigma))).T for i in range(a1 - a0)]
    wdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('Waveform', np.int16, window)])
    waves = np.empty(a1 - a0).astype(wdtp)
    pan = np.arange(window)
    for i in range(a1 - a0):
        wave = np.sum([np.where(pan > sams[i][j, 0], wff.spe(pan - sams[i][j, 0], tau=p[0], sigma=p[1], A=p[2]) * sams[i][j, 1] / gmu, 0) for j in range(len(sams[i]))], axis=0)
        if args.noi:
            wave = wave + np.random.normal(0, std, size=window)
        waves[i]['Waveform'] = np.around(wave).astype(np.int16)
    tdtp = np.dtype([('TriggerNo', np.uint32), ('T0', np.float64)])
    t = np.empty(a1 - a0).astype(tdtp)
    t['TriggerNo'] = np.arange(a0, a1).astype(np.uint32)
    t['T0'] = t0
    waves['TriggerNo'] = np.arange(a0, a1).astype(np.uint32)
    waves['ChannelID'] = 0
    sdtp = np.dtype([('TriggerNo', np.uint32), ('PMTId', np.uint32), ('HitPosInWindow', np.float64), ('Charge', np.float64)])
    samples = np.empty(sum([len(sams[i]) for i in range(a1 - a0)])).astype(sdtp)
    samples['TriggerNo'] = np.repeat(np.arange(a0, a1), [len(sams[i]) for i in range(a1 - a0)]).astype(np.uint32)
    samples['PMTId'] = 0
    samples['HitPosInWindow'] = np.hstack([sams[i][:, 0] for i in range(a1 - a0)])
    samples['Charge'] = np.hstack([sams[i][:, 1] for i in range(a1 - a0)])
    return t, samples, waves

chunk = args.N // args.Ncpu + 1
slices = np.vstack((np.arange(0, args.N, chunk), np.append(np.arange(chunk, args.N, chunk), args.N))).T.astype(np.int).tolist()

with Pool(min(Ncpu, cpu_count())) as pool:
    result = pool.starmap(partial(sampling, mu=Mu, tau=Tau, sigma=Sigma), slices)

t0 = np.hstack([result[i][0] for i in range(len(result))])
samples = np.hstack([result[i][1] for i in range(len(result))])
waves = np.hstack([result[i][2] for i in range(len(result))])

v = np.logical_not(np.any(np.isnan(waves['Waveform']), axis=1) | np.isin(waves['TriggerNo'], samples['TriggerNo'][samples['HitPosInWindow'] > window - 1]))
if np.sum(v) != args.N:
    t0 = t0[v]
    samples = samples[np.isin(samples['TriggerNo'], waves['TriggerNo'][v])]
    waves = waves[v]

assert not np.all(np.isnan(waves['Waveform']))

with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('SimTruth/T', data=t0, compression='gzip', compression_opts=4)
    opt.create_dataset('SimTriggerInfo/PEList', data=samples, compression='gzip', compression_opts=4)
    dset = opt.create_dataset('Readout/Waveform', data=waves, compression='gzip', compression_opts=4)
    dset.attrs['mu'] = Mu
    dset.attrs['tau'] = Tau
    dset.attrs['sigma'] = Sigma
print(args.opt + ' saved, l =', int(np.sum(v)))

if not os.path.exists('spe.h5'):
    with h5py.File('spe.h5', 'w') as spp:
        dset = spp.create_dataset('SinglePE', data=[])
        dset.attrs['SpePositive'] = wff.spe(np.arange(80), p[0], p[1], p[2])[np.newaxis, ...]
        dset.attrs['Epulse'] = 1
        dset.attrs['Std'] = [std]
        dset.attrs['ChannelID'] = [0]
        dset.attrs['parameters'] = [p]
    print('spe.h5 saved')