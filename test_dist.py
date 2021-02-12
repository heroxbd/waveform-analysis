import sys
import numpy as np
import csv
import h5py
import scipy.stats
import itertools as it
import argparse
from multiprocessing import Pool, cpu_count
import wf_func as wff

psr = argparse.ArgumentParser()
psr.add_argument('--ref', dest='ref', help='reference file', nargs='+')
psr.add_argument('ipt', help='input file')
psr.add_argument('-N', dest='Ncpu', type=int, help='cpu number', default=50)
psr.add_argument('-o', dest='opt', help='output file')
args = psr.parse_args()

fref = args.ref[0]
fipt = args.ipt
fopt = args.opt
Ncpu = args.Ncpu

def wpdist(a, b):
    dt = np.zeros(b - a, dtype=opdt)
    dt['chargediff'] = np.nan
    pan = np.arange(window)
    for i, c in zip(range(a, b), range(b - a)):
        cid = df_wav[i_wav[i]]['ChannelID']
        p = spe_pre[cid]['parameters']
        wave = df_wav[i_wav[i]]['Waveform'].astype(np.float64) * spe_pre[cid]['epulse']
        
        pet_sub = df_sub[i_sub[i]:i_sub[i+1]]['HitPosInWindow']
        cha_sub = df_sub[i_sub[i]:i_sub[i+1]]['Charge']
        wav_sub = np.sum([np.where(pan > pet_sub[j], wff.spe(pan - pet_sub[j], tau=p[0], sigma=p[1], A=p[2]) * cha_sub[j], 0) for j in range(len(pet_sub))], axis=0)
        pet_ans_0 = df_ans[i_ans[i]:i_ans[i+1]]['HitPosInWindow']
        cha_ans = df_ans[i_ans[i]:i_ans[i+1]]['Charge']
        pet_ans = np.unique(pet_ans_0)
        cha_ans = np.array([np.sum(cha_ans[pet_ans_0 == j]) for j in pet_ans])
        wav_ans = np.sum([np.where(pan > pet_ans[j], wff.spe(pan - pet_ans[j], tau=p[0], sigma=p[1], A=p[2]) * cha_ans[j], 0) for j in range(len(pet_ans))], axis=0)
        wav_ans = wav_ans / np.sum(spe_pre[cid]['spe'])
        wav_sub = wav_sub / np.sum(spe_pre[cid]['spe'])
        dt['chargediff'][c] = np.sum(cha_sub) - np.sum(cha_ans)
        dt['NPE'][c] = len(pet_ans)

        dt['wdist'][c] = scipy.stats.wasserstein_distance(pet_ans, pet_sub, u_weights=cha_ans, v_weights=cha_sub)
        dt['TriggerNo'][c] = df_wav[i_wav[i]]['TriggerNo']
        dt['ChannelID'][c] = cid
        dt['RSS'][c] = np.power(wav_sub - wav_ans, 2).sum()
        dt['RSS_truth'][c] = np.power(wav_ans - wave, 2).sum()
        dt['RSS_recon'][c] = np.power(wav_sub - wave, 2).sum()
    return dt

spe_pre = wff.read_model(args.ref[1])
with h5py.File(fref, 'r', libver='latest', swmr=True) as ref, h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
    df_ans = ref['SimTriggerInfo']['PEList'][:]
    df_wav = ref['Readout']['Waveform'][:]
    df_sub = ipt['photoelectron'][:]
    method = ipt['photoelectron'].attrs['Method']
    Mu = ipt['photoelectron'].attrs['mu']
    Tau = ipt['photoelectron'].attrs['tau']
    Sigma = ipt['photoelectron'].attrs['sigma']
df_ans = np.sort(df_ans, kind='stable', order=['TriggerNo', 'PMTId'])
df_sub = np.sort(df_sub, kind='stable', order=['TriggerNo', 'ChannelID'])
df_wav = np.sort(df_wav, kind='stable', order=['TriggerNo', 'ChannelID'])
Chnum = len(np.unique(df_ans['PMTId']))
e_ans = df_ans['TriggerNo'] * Chnum + df_ans['PMTId']
e_ans, i_ans = np.unique(e_ans, return_index=True)
i_ans = np.append(i_ans, len(df_ans))

opdt = np.dtype([('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('NPE', np.uint16), ('wdist', np.float64), ('chargediff', np.float64), ('RSS', np.float64), ('RSS_recon', np.float64), ('RSS_truth', np.float64)])
window = len(df_wav[0]['Waveform'])

e_wav = df_wav['TriggerNo'] * Chnum + df_wav['ChannelID']
df_wav = df_wav[np.isin(e_wav, e_ans)]
e_wav, i_wav = np.unique(df_wav['TriggerNo'] * Chnum + df_wav['ChannelID'], return_index=True)

e_sub = df_sub['TriggerNo']*Chnum + df_sub['ChannelID']
df_sub = df_sub[np.isin(e_sub, e_ans)]
e_sub, i_sub = np.unique(df_sub['TriggerNo'] * Chnum + df_sub['ChannelID'], return_index=True)
i_sub = np.append(i_sub, len(df_sub))
assert len(e_ans) ==  len(e_wav) and len(e_ans) == len(e_sub), 'Incomplete Submission'

l = len(e_sub)
chunk = l // Ncpu + 1
slices = np.vstack((np.arange(0, l, chunk), np.append(np.arange(chunk, l, chunk), l))).T.astype(int).tolist()
with Pool(min(Ncpu, cpu_count())) as pool:
    result = pool.starmap(wpdist, slices)
dt = np.hstack(result)
with h5py.File(fopt, 'w') as h5f:
    dset = h5f.create_dataset('Record', data=dt, compression='gzip')
    dset.attrs['Method'] = method
    dset.attrs['mu'] = Mu
    dset.attrs['tau'] = Tau
    dset.attrs['sigma'] = Sigma