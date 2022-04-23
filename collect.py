import argparse
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

import wf_func as wff

TRIALS = wff.TRIALS

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--mu', type=str, help='mu file')
psr.add_argument('--sparse', type=str, help='LucyDDM file')
psr.add_argument('--ref', type=str, help='truth file')
psr.add_argument('-N', '--Ncpu', dest='Ncpu', type=int, default=25)
args = psr.parse_args()

# sample = pd.read_hdf(args.ipt, 'sample').set_index(['TriggerNo', 'ChannelID'])
# s_max = pd.read_hdf(args.ipt, 's_max').set_index(['TriggerNo', 'ChannelID'])
# index = pd.read_hdf(args.sparse, 'index').set_index(['TriggerNo', 'ChannelID'])
# mu = pd.read_hdf(args.mu, 'mu').set_index(['TriggerNo', 'ChannelID'])

# np.sort(, kind='stable', order=['TriggerNo', 'ChannelID'])

with h5py.File(args.ipt, 'r', libver='latest', swmr=True) as ipt:
    s0 = ipt['sample']['s0'][:]
    s_max = ipt['s_max'][:]

with h5py.File(args.mu, 'r', libver='latest', swmr=True) as ipt:
    mu = ipt['mu'][:]

with h5py.File(args.sparse, 'r', libver='latest', swmr=True) as ipt:
    A = ipt['A'][:]
    cx = ipt['cx'][:]
    index = ipt['index'][:]
    s = ipt['s'][:]
    tq = ipt['tq'][:]
    z = ipt['z'][:]

l_e = len(index)

with h5py.File(args.ref, 'r', libver='latest', swmr=True) as ipt:
    ent = ipt['Readout/Waveform'][:]
    N = len(ent)
    print('{} waveforms are computed'.format(N))
    Mu = ipt['Readout/Waveform'].attrs['mu'].item()
    Tau = ipt['Readout/Waveform'].attrs['tau'].item()
    Sigma = ipt['Readout/Waveform'].attrs['sigma'].item()
    gmu = ipt['SimTriggerInfo/PEList'].attrs['gmu'].item()

s_index = s0[s_max['s_max_index'] + np.arange(len(s_max['s_max'])) * TRIALS]
opdt = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), 
                 ('HitPosInWindow', np.float64), ('Charge', np.float64)])
sdtp = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), 
                 ('tscharge', np.float64), ('tswave', np.float64), 
                 ('mucharge', np.float64), ('muwave', np.float64), 
                 ('consumption', np.float64)])


def collect(a0, a1):
    start = 0
    dt = np.zeros(s_index.sum(), dtype=opdt)
    ts = np.zeros(a1 - a0, dtype=sdtp)
    for i in range(a0, a1):
        l_t = index['l_t'][i]
        tlist = tq['t_s'][i][:l_t]
        mus = index['mus'][i]
        sig2s = index['sig2s'][i]
        sig2w = index['sig2w'][i]
        y = ent[i]['Waveform'][index['a_wave'][i]:index['b_wave'][i]]
        A_i = A[i][:index['b_wave'][i]-index['a_wave'][i], :l_t]

        s = s_max['s_max'][i][:s_index[i]]
        t, c = np.unique(np.sort(np.digitize(s, bins=np.arange(l_t)) - 1), return_counts=True)
        c_star = np.zeros(l_t, dtype=int)
        c_star[t] = c
        zx = y - np.dot(A_i, mus * c_star)
        Phi_s = wff.Phi(y, A_i, c_star, mus, sig2s, sig2w)
        invPhi = np.linalg.inv(Phi_s)
        xmmse_most = mus * c_star + np.matmul(np.diagflat(sig2s * c_star), np.matmul(A_i.T, np.matmul(invPhi, zx)))
        pet = np.repeat(tlist[xmmse_most > 0], c_star[xmmse_most > 0])
        cha = np.repeat(xmmse_most[xmmse_most > 0] / mus / c_star[xmmse_most > 0], c_star[xmmse_most > 0])
        mu_i = (c_star > 0).sum()
        t0_i, _ = wff.likelihoodt0(pet, char=cha, gmu=gmu, Tau=Tau, Sigma=Sigma, mode='all')
        pet, cha = wff.clip(pet, cha, 0.0)
        cha = cha * gmu
        end = start + len(cha)

        ts['muwave'][i - a0] = mu['mu'][i]
        # ts['tswave'][i - a0] = mu['t0'][i] # use Gibbs sampled t0
        ts['tswave'][i - a0] = s_max['t0'][i] # use MLE fitted t0 
        ts['mucharge'][i - a0] = mu_i
        ts['tscharge'][i - a0] = t0_i
        ts['TriggerNo'][i - a0] = ent[i]['TriggerNo']
        ts['ChannelID'][i - a0] = ent[i]['ChannelID']
        ts['consumption'][i - a0] = s_max[i]['consumption']

        dt['HitPosInWindow'][start:end] = pet
        dt['Charge'][start:end] = cha
        dt['TriggerNo'][start:end] = ent[i]['TriggerNo']
        dt['ChannelID'][start:end] = ent[i]['ChannelID']
        start = end
    dt = np.sort(dt[:end], kind='stable', order=['TriggerNo', 'ChannelID'])
    return dt, ts

if args.Ncpu == 1:
    slices = [[0, N]]
else:
    chunk = N // args.Ncpu + 1
    slices = np.vstack((np.arange(0, N, chunk), np.append(np.arange(chunk, N, chunk), N))).T.astype(int).tolist()

collect(0, 200)
with Pool(min(args.Ncpu, cpu_count())) as pool:
    result = pool.starmap(partial(collect), slices)

dt = np.hstack([result[i][0] for i in range(len(slices))])
ts = np.hstack([result[i][1] for i in range(len(slices))])

with h5py.File(args.opt, 'w') as opt:
    pedset = opt.create_dataset('photoelectron', data=dt, compression='gzip')
    pedset.attrs['Method'] = 'fsmp'
    pedset.attrs['mu'] = Mu
    pedset.attrs['tau'] = Tau
    pedset.attrs['sigma'] = Sigma
    tsdset = opt.create_dataset('starttime', data=ts, compression='gzip')
    print('The output file path is {}'.format(args.opt))
