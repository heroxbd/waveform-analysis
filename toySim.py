import os
import itertools
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

import h5py
from tqdm import tqdm
from tqdm import trange
import numpy as np
from numpy.lib import recfunctions
# np.seterr(all='raise')
import scipy.optimize as optimize
import scipy.interpolate as interpolate
from scipy.stats import poisson, uniform, norm, chi2
from scipy import integrate
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

window = wff.window
gmu = wff.gmu
gsigma = wff.gsigma
mtslist = args.mts.split('-')
Mu = float(mtslist[0])
Tau = float(mtslist[1])
Sigma = float(mtslist[2])

p = wff.p
p[2] = p[2] * gmu / integrate.quad(lambda t : wff.spe(np.array([t]), tau=p[0], sigma=p[1], A=p[2]), 0, 100)[0]
std = wff.std

def sampling(a0, a1, mu, tau, sigma):
    np.random.seed(a0 + round(Tau + Sigma))
    npe = poisson.ppf(1 - uniform.rvs(scale=1-poisson.cdf(0, mu), size=a1 - a0), mu).astype(int)
    t0 = np.random.uniform(100., 500., size=a1 - a0)
    sams = [np.vstack((wff.time(npe[i], tau, sigma) + t0[i], wff.charge(npe[i], gmu=gmu, gsigma=gsigma, thres=0))).T for i in range(a1 - a0)]
    # sams = [np.vstack((np.arange(npe[i]) + t0[i], wff.charge(npe[i], gmu=gmu, gsigma=gsigma, thres=0))).T for i in range(a1 - a0)]
    wdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('Npe', np.uint), ('Waveform', np.float, window * wff.nshannon)])
    waves = np.empty(a1 - a0).astype(wdtp)
    pan = np.arange(0, window, 1 / wff.nshannon)
    for i in range(a1 - a0):
        wave = np.sum([wff.spe((pan - sams[i][j, 0] + np.abs(pan - sams[i][j, 0])) / 2, tau=p[0], sigma=p[1], A=p[2]) * sams[i][j, 1] / gmu for j in range(len(sams[i]))], axis=0)
        if args.noi:
            wave = wave + np.random.normal(0, std, size=window * wff.nshannon)
        waves[i]['Waveform'] = wave
    tdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('T0', np.float64)])
    t = np.empty(a1 - a0).astype(tdtp)
    t['TriggerNo'] = np.arange(a0, a1).astype(np.uint32)
    t['T0'] = t0
    t['ChannelID'] = 0
    waves['TriggerNo'] = np.arange(a0, a1).astype(np.uint32)
    waves['ChannelID'] = 0
    waves['Npe'] = npe
    sdtp = np.dtype([('TriggerNo', np.uint32), ('PMTId', np.uint32), ('HitPosInWindow', np.float64), ('Charge', np.float64)])
    pelist = np.empty(sum([len(sams[i]) for i in range(a1 - a0)])).astype(sdtp)
    pelist['TriggerNo'] = np.repeat(np.arange(a0, a1), [len(sams[i]) for i in range(a1 - a0)]).astype(np.uint32)
    pelist['PMTId'] = 0
    pelist['HitPosInWindow'] = np.hstack([sams[i][:, 0] for i in range(a1 - a0)])
    pelist['Charge'] = np.hstack([sams[i][:, 1] for i in range(a1 - a0)])
    return t, pelist, waves

chunk = args.N // args.Ncpu + 1
slices = np.vstack((np.arange(0, args.N, chunk), np.append(np.arange(chunk, args.N, chunk), args.N))).T.astype(int).tolist()

with Pool(min(args.Ncpu, cpu_count())) as pool:
    result = pool.starmap(partial(sampling, mu=Mu, tau=Tau, sigma=Sigma), slices)

t0 = np.hstack([result[i][0] for i in range(len(result))])
pelist = np.hstack([result[i][1] for i in range(len(result))])
waves = np.hstack([result[i][2] for i in range(len(result))])

Chnum = len(np.unique(pelist['PMTId']))
e_pel = pelist['TriggerNo'] * Chnum + pelist['PMTId']
e_pel, i_pel = np.unique(e_pel, return_index=True)
i_pel = np.append(i_pel, len(pelist))

def start_time(a0, a1):
    stime = np.empty(a1 - a0)
    for i in range(a0, a1):
        hitt = pelist[i_pel[i]:i_pel[i+1]]['HitPosInWindow'].astype(np.float64)
        t0, _ = wff.likelihoodt0(hitt, char=None, gmu=gmu, Tau=Tau, Sigma=Sigma, mode='all')
        stime[i - a0] = t0
    return stime

sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('ts1sttruth', np.float64), ('tstruth', np.float64)])
ts = np.zeros(args.N, dtype=sdtp)
with Pool(min(args.Ncpu, cpu_count())) as pool:
    result = pool.starmap(partial(start_time), slices)
ts['TriggerNo'] = waves['TriggerNo']
ts['ChannelID'] = waves['ChannelID']
ts['tstruth'] = np.hstack(result)
ts['ts1sttruth'] = np.array([np.min(pelist[i_pel[i]:i_pel[i+1]]['HitPosInWindow']) for i in range(len(e_pel))])
t0 = recfunctions.join_by(('TriggerNo', 'ChannelID'), ts, t0, usemask=False)

vali = np.full(len(waves), False)
valit = []
valit.append(np.sum(waves['Waveform'], axis=1) <= 0)
valit.append(np.any(np.isnan(waves['Waveform']), axis=1))
valit.append(np.isin(waves['TriggerNo'], pelist['TriggerNo'][pelist['HitPosInWindow'] > window - 1]))
r = 7
valit.append(np.abs(t0['tstruth'] - t0['T0'] - np.mean(t0['tstruth'] - t0['T0'])) > r * np.std(t0['tstruth'] - t0['T0'], ddof=-1))
valit.append(np.sum(waves['Waveform'] > 5 * std, axis=1) <= 0)
valit.append(np.isin(waves['TriggerNo'], pelist['TriggerNo'][pelist['Charge'] <= 0]) & np.isin(waves['ChannelID'], pelist['PMTId'][pelist['Charge'] <= 0]))
for i, t in enumerate(valit):
    print(f'Loss {i} is {t.sum()}, {t.sum()/len(t):.2%}')
    vali |= t
npe_removed = waves['Npe'][vali]
pelist_removed = pelist[np.isin(pelist['TriggerNo'], waves['TriggerNo']) & np.isin(pelist['PMTId'], waves['ChannelID'])]
wavesum_removed = np.empty(vali.sum())
for i, w in enumerate(waves[vali]):
    wavesum_removed[i] = pelist_removed['Charge'][(pelist_removed['TriggerNo'] == w['TriggerNo']) & (pelist_removed['PMTId'] == w['ChannelID'])].sum()

vali = np.logical_not(vali)
if np.sum(vali) != args.N:
    t0 = t0[vali]
    waves = waves[vali]
    pelist = pelist[np.isin(pelist['TriggerNo'], waves['TriggerNo'])]

assert not np.all(np.isnan(waves['Waveform']))

with h5py.File(args.opt, 'w') as opt:
    dset = opt.create_dataset('SimTruth/T', data=t0, compression='gzip', compression_opts=4)
    dset.attrs['r'] = r
    dset = opt.create_dataset('SimTriggerInfo/PEList', data=pelist, compression='gzip', compression_opts=4)
    dset.attrs['gmu'] = gmu
    dset.attrs['gsigma'] = gsigma
    dset = opt.create_dataset('Readout/Waveform', data=waves, compression='gzip', compression_opts=4)
    dset.attrs['mu'] = Mu
    dset.attrs['tau'] = Tau
    dset.attrs['sigma'] = Sigma
    dset.attrs['Std'] = std
    dset.attrs['npe_removed'] = npe_removed
    dset.attrs['wavesum_removed'] = wavesum_removed
print(args.opt + ' saved, l =', int(np.sum(vali)))

if not os.path.exists('spe.h5'):
    with h5py.File('spe.h5', 'w') as spp:
        dset = spp.create_dataset('SinglePE', data=[])
        dset.attrs['SpePositive'] = wff.spe(np.arange(80), p[0], p[1], p[2])[np.newaxis, ...]
        dset.attrs['Epulse'] = 1
        dset.attrs['Std'] = [std]
        dset.attrs['ChannelID'] = [0]
        dset.attrs['parameters'] = [p]
    print('spe.h5 saved')
