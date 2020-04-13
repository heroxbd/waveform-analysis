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

R = 4

class Ind_Generator:
    def __init__(self, sigma, l):
        self.sigma = sigma
        self.l = l
        np.random.seed(0)
        return

    def next_ind(self, ind):
        while True:
            ind_n = self.sigma * np.random.randn() + (ind+0.5)
            if ind_n >= 0 or ind_n < self.l:
                ind_n = np.int(ind_n).astype(np.uint16)
                break
        return ind_n

    def uni_rand(self):
        return np.random.rand()

def main(fopt, fipt, single_pe_path):
    epulse = wfaf.estipulse(fipt)
    spemean = wfaf.generate_model(single_pe_path, epulse)
    spemean = epulse * spemean
    _, _, m_l, _, _, thres = wfaf.pre_analysis(fipt, epulse, spemean)
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
        L = Length_pe - len(spemean) + 1
        sigma = L / (2*R)
        gen = Ind_Generator(sigma, L)
        N = 11000
        panel = np.zeros(L)
        for i in range(l):
            wf_input = ent[i]['Waveform']
            wave = epulse * wfaf.deduct_base(-1*epulse*wf_input, m_l, thres, 10, 'detail')
            sampling = np.zeros(N).astype(np.uint16)
            r_a = np.power(np.convolve(panel, spemean) - wave, 2)
            for j in range(1, N):
                u = gen.uni_rand()
                ind = gen.next_ind(sampling[j - 1])
                panel_a = panel
                panel_a[ind] = 1
                r_b = np.power(np.convolve(panel_a, spemean) - wave, 2)
                v = np.min(1, r_a/r_b)
                if u < v:
                    sampling[j] = ind
                else:
                    sampling[j] = sampling[j-1]
                r_a = r_b
            sampling = sampling[1000:]
            pwe_tot = np.sum(wave) / np.sum(spemean)
            pet, pwe = np.unique(sampling, return_counts=True)
            pwe = pwe / pwe_tot

            if np.max(pwe) < 0.1:
                t = np.where(wave == wave.min())[0][:1] - np.argmin(spemean)
                pet = t if t[0] >= 0 else np.array([0])
                pwe = np.array([1])
            pet = pet[pwe > 0.1]
            pwe = pwe[pwe > 0.1].astype(np.float16)
            lenpf = len(pwe)
            end = start + lenpf
            dt['PETime'][start:end] = pet
            dt['Weight'][start:end] = pwe
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