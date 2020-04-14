# -*- coding: utf-8 -*-

import sys
sys.path.append('test')
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import wf_analysis_func as wfaf

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output file')
psr.add_argument('ipt', help='input file')
psr.add_argument('--ref', dest='ref', help='reference file')
psr.add_argument('-p', dest='print', action='store_false', help='print bool', default=True)
args = psr.parse_args()

if args.print:
    sys.stdout = None

def lucyDDM(waveform, spe, iterations=50):
    '''Lucy deconvolution
    Parameters
    ----------
    waveform : 1d array
    spe : 1d array
        point spread function; single photon electron response
    iterations : int

    Returns
    -------
    signal : 1d array

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] https://github.com/scikit-image/scikit-image/blob/master/skimage/restoration/deconvolution.py#L329
    '''
    waveform = waveform + 0.001
    spe = spe + 0.001
    waveform = waveform / np.sum(spe)
    # use the deconvlution method
    wave_deconv = np.full(waveform.shape, 0.5)
    spe_mirror = spe[::-1]
    for _ in range(iterations):
        relative_blur = waveform / np.convolve(wave_deconv, spe, 'same')
        wave_deconv = wave_deconv * np.convolve(relative_blur, spe_mirror, 'same')
        # there is no need to set the bound if the spe and the wave are all none negative 
    return wave_deconv

def main(fopt, fipt, single_pe_path):
    spemean_r, epulse = wfaf.generate_model(single_pe_path)
    spe_pre = wfaf.pre_analysis(fipt, epulse, -1*epulse*spemean_r)
    spemean = spe_pre['epulse'] * spe_pre['spemean']
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
        ent = ipt['Waveform']
        Length_pe = len(ent[0]['Waveform'])
        assert Length_pe >= len(spemean), 'Single PE too long which is {}'.format(len(spemean))
        spemean = np.concatenate([spemean, np.zeros(Length_pe - len(spemean))])
        l = len(ent)
        print('{} waveforms will be computed'.format(l))
        dt = np.zeros(l * (Length_pe//5), dtype=opdt)
        start = 0
        end = 0
        for i in range(l):
            wf_input = ent[i]['Waveform']
            wave = spe_pre['epulse'] * wfaf.deduct_base(-1*spe_pre['epulse']*wf_input, spe_pre['m_l'], spe_pre['thres'], 10, 'detail')
            wave = np.where(wave < 0, 0, wave)
            pf = lucyDDM(wave, spemean, 50)

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
