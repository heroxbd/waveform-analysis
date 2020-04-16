# -*- coding: utf-8 -*-

import re
import numpy as np
from scipy import optimize as opti
import h5py
import sys
sys.path.append('test')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import argparse
import wf_analysis_func as wfaf

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('ipt', help='input file')
psr.add_argument('--ref', dest='ref', help='reference file')
psr.add_argument('--num', type=int, help='fragment number')
psr.add_argument('-p', dest='print', action='store_false', help='print bool', default=True)
args = psr.parse_args()

if args.print:
    sys.stdout = None

N = 1100

class Ind_Generator:
    def __init__(self, sigma):
        self.sigma = sigma
        np.random.seed(0)
        return

    def next_ind(self, ind):
        ind_n = np.zeros_like(ind)
        for i in range(len(ind)):
            n = self.sigma * np.random.randn() + ind[i]
            if n > 0:
                ind_n[i] = n
            else:
                ind_n[i] = 0
        return ind_n

    def uni_rand(self):
        return np.random.rand()

def mcmc_N(wave, spemean, gen):
    L = len(wave) - len(spemean) + 1
    sampling = np.zeros((N, L))
    sampling[0] = 0.1
    r_a = np.sum(np.power(np.convolve(sampling[0], spemean) - wave, 2))
    for j in range(N):
        u = gen.uni_rand()
        ind = gen.next_ind(sampling[j - 1])
        r_b = np.sum(np.power(np.convolve(ind, spemean) - wave, 2))
        v = np.min([1, r_a/r_b])
        if u < v:
            sampling[j] = ind
        else:
            sampling[j] = sampling[j-1]
        r_a = r_b
    sampling = sampling[N//10:]
    pf = np.zeros_like(wave)
    pf[:L] = np.mean(sampling, axis=0)
    return pf

def main(fopt, fipt, single_pe_path):
    spemean_r, epulse = wfaf.generate_model(single_pe_path)
    spe_pre = wfaf.pre_analysis(fipt, epulse, -1*epulse*spemean_r)
    spemean = epulse * spe_pre['spemean']
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        lenfr = math.floor(len(ent)/(args.num+1))
        num = int(re.findall(r'-\d+\.h5', fopt, flags=0)[0][1:-3])
        if (num+2)*lenfr > len(ent):
            ent = ent[num*lenfr:]
        else:
            ent = ent[num*lenfr:(num+1)*lenfr]

        Length_pe = len(ent[0]['Waveform'])
        assert Length_pe >= len(spemean), 'Single PE too long which is {}'.format(len(spemean))
        l = len(ent)
        print('{} waveforms will be computed'.format(l))
        dt = np.zeros(l * (Length_pe//5), dtype=opdt)
        start = 0
        end = 0
        gen = Ind_Generator(0.1)

        for i in range(l):
            wf_input = ent[i]['Waveform']
            wave = epulse * wfaf.deduct_base(-1*epulse*wf_input, spe_pre['m_l'], spe_pre['thres'], 10, 'detail')
            pf = mcmc_N(wave, spemean, gen)
            pet, pwe = wfaf.pf_to_tw(pf, 0.1)

            lenpf = len(pwe)
            end = start + lenpf
            dt['PETime'][start:end] = pet.astype(np.uint16)
            dt['Weight'][start:end] = pwe.astype(np.float16)
            dt['EventID'][start:end] = ent[i]['EventID']
            dt['ChannelID'][start:end] = ent[i]['ChannelID']
            start = end
            print('\rAnsw Generating:|{}>{}|{:6.2f}%'.format(((20*i)//l)*'-', (19-(20*i)//l)*' ', 100 * ((i+1) / l)), end='' if i != l-1 else '\n')
    dt = dt[dt['Weight'] > 0]
    dt = np.sort(dt, kind='stable', order=['EventID', 'ChannelID', 'PETime'])
    with h5py.File(fopt, 'w') as opt:
        opt.create_dataset('Answer', data=dt, compression='gzip')
        print('The output file path is {}'.format(fopt), end=' ', flush=True)
    return

if __name__ == '__main__':
    main(args.opt, args.ipt, args.ref)
