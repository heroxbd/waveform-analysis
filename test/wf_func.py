# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import optimize as opti
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

class Ind_Generator:
    def __init__(self):
        # self.sigma = sigma
        np.random.seed(0)
        return

    def next_ind(self, ind):
        ind_n = ind.copy()
        for i in range(len(ind)):
            v = np.random.rand()
            if v < 1/3:
                ind_n[i] = ind[i] - 1
            elif v >= 2/3:
                ind_n[i] = ind[i] + 1
        return ind_n

    def uni_rand(self):
        return np.random.rand()

def fit_N(wave, spe_pre, method, *args):
    l = wave.shape[0]
    lowp = np.argwhere(vali_base(wave, spe_pre['m_l'], spe_pre['thres']) == 1).flatten()
    lowp = lowp[np.logical_and(lowp > 1, lowp < l-1)]
    flag = 1
    if len(lowp) != 0:
        panel = np.zeros(l)
        for j in lowp:
            head = j-spe_pre['mar_l'] if j-spe_pre['mar_l'] > 0 else 0
            tail = j+spe_pre['mar_r'] if j+spe_pre['mar_r'] <= l else l
            panel[head : tail + 1] = 1
        fitp = np.argwhere(panel == 1).flatten()
        numb = np.argwhere(wave[lowp+1]-wave[lowp]-wave[lowp-1]+wave[lowp-2] > 1.5).flatten()
        if len(numb) != 0:
            ran = np.arange(spe_pre['peak_c'] - 1, spe_pre['peak_c'] + 1)
            possible = np.unique(lowp[numb] - ran.reshape(ran.shape[0], 1))
            possible = possible[np.logical_and(possible>=0, possible<l)]
            if len(possible) != 0:
                if method == 'xiaopeip':
                    pf_r = xiaopeip_core(wave, spe_pre['spe'], fitp, possible)
                elif method == 'mcmc':
                    pf_r = mcmc_core(wave, spe_pre['spe'], fitp, possible, *args)
            else:
                flag = 0
        else:
            flag = 0
    else:
        flag = 0
    if flag == 0:
        t = np.where(wave == wave.min())[0][:1] - spe_pre['peak_c']
        possible = t if t[0] >= 0 else np.array([0])
        pf_r = np.array([1])
    pf = np.zeros_like(wave)
    pf[possible] = pf_r
    return pf, fitp, possible

def xiaopeip_core(wave, spe, fitp, possible):
    l = wave.shape[0]
    spe = np.concatenate([spe, np.zeros(l - spe.shape[0])])
    norm_fit = lambda x, M, p: np.linalg.norm(p - np.matmul(M, x))
    ans0 = np.zeros_like(possible).astype(np.float64)
    b = np.zeros((possible.shape[0], 2)).astype(np.float64)
    b[:, 1] = np.inf
    mne = spe[np.mod(fitp.reshape(fitp.shape[0], 1) - possible.reshape(1, possible.shape[0]), l)]
    ans = opti.fmin_l_bfgs_b(norm_fit, ans0, args=(mne, wave[fitp]), approx_grad=True, bounds=b, maxfun=500000)
    # ans = opti.fmin_slsqp(norm_fit, ans0, args=(mne, wave[fitp]), bounds=b, iprint=-1, iter=500000)
    # ans = opti.fmin_tnc(norm_fit, ans0, args=(mne, wave[fitp]), approx_grad=True, bounds=b, messages=0, maxfun=500000)
    pf_r = ans[0]
    return pf_r

def lucyddm_core(waveform, spe, iterations=100):
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
    wave = np.where(waveform > 0, waveform, 0)
    wave = wave + 0.001
    wave = wave / np.sum(spe)
    t = np.argwhere(spe > 0)[0][0]
    spe_t = spe[spe > 0]
    l = spe_t.shape[0]
    # use the deconvlution method
    wave_deconv = np.full(wave.shape, 0.1)
    spe_mirror = spe_t[::-1]
    for _ in range(iterations):
        relative_blur = wave / np.convolve(wave_deconv, spe_t, 'same')
        wave_deconv = wave_deconv * np.convolve(relative_blur, spe_mirror, 'same')
        # there is no need to set the bound if the spe and the wave are all none negative
    wave_deconv = np.append(wave_deconv[(l-1)//2+t:], np.zeros((l-1)//2+t))
    # np.convolve(wave_deconv, spe, 'full')[:len(wave)] should be wave
    wave_deconv = np.where(wave_deconv<50, wave_deconv, 0)
    return wave_deconv

def mcmc_core(wave, spe, fitp, possible, *args):
    gen = args[0]
    num = 1000
    l = wave.shape[0]
    spe = np.concatenate([spe, np.zeros(l - spe.shape[0])])
    likelihood = lambda x, M, p: 1000 / np.sum(np.abs(p - np.matmul(M, x)))
    prior = lambda x: np.all(x >= 0).astype(np.float)
    mne = spe[np.mod(fitp.reshape(fitp.shape[0], 1) - possible.reshape(1, possible.shape[0]), l)]
    samp = np.zeros((num, possible.shape[0]))
    samp[0] = 1
    like_a = likelihood(samp[0], mne, wave[fitp])
    for j in range(1, num):
        u = gen.uni_rand()
        ind = gen.next_ind(samp[j - 1])
        like_b = likelihood(ind, mne, wave[fitp])
        if u < (like_b*prior(ind))/(like_a*prior(samp[j - 1])):
            samp[j] = ind
            like_a = like_b
        else:
            samp[j] = samp[j - 1]
    samp = samp[num//10:]
    pf = np.mean(samp, axis=0)
    return pf

def xpp_convol(pet, wgt):
    core = np.array([0.9, 1.7, 0.9])
    idt = np.dtype([('PETime', np.int16), ('Weight', np.float16), ('Wgt_b', np.uint8)])
    seg = np.zeros(np.max(pet) + 3, dtype=idt)
    seg['PETime'] = np.arange(-1, np.max(pet) + 2)
    seg['Weight'][np.sort(pet) + 1] = wgt[np.argsort(pet)]
    seg['Wgt_b'] = np.around(seg['Weight'])
    resi = seg['Weight'][1:-1] - seg['Wgt_b'][1:-1]
    t = np.convolve(resi, core, 'full')
    ta = np.diff(t, prepend=t[0])
    tb = np.diff(t, append=t[-1])
    seg['Wgt_b'][(ta > 0)*(tb < 0)*(t > 0.5)*(seg['Wgt_b'] == 0.0)*(seg['Weight'] > 0)] += 1
    if np.sum(seg['Wgt_b'][1:-1] > 0) != 0:
        pwe = seg['Wgt_b'][1:-1][seg['Wgt_b'][1:-1] > 0]
        pet = seg['PETime'][1:-1][seg['Wgt_b'][1:-1] > 0]
    else:
        pwe = np.array([1])
        pet = seg['PETime'][np.argmax(seg['Weight'])]
    return pet, pwe

def read_model(spe_path):
    with h5py.File(spe_path, 'r', libver='latest', swmr=True) as speFile:
        spe = speFile['SinglePE'].attrs['SpePositive']
        epulse = speFile['SinglePE'].attrs['Epulse']
        thres = speFile['SinglePE'].attrs['Thres']
        m_l = np.sum(spe > thres)
        peak_c = np.argmax(spe)
        mar_l = np.sum(spe[:peak_c] < thres)
        mar_r = np.sum(spe[peak_c:] < thres)
    spe_pre = {'spe':spe, 'epulse':epulse, 'peak_c':peak_c, 'm_l':m_l, 'mar_l':mar_l, 'mar_r':mar_r, 'thres':thres}
    return spe_pre

def snip_baseline(waveform, itera=20):
    wm = np.min(waveform)
    waveform = waveform - wm
    v = np.log(np.log(np.sqrt(waveform+1)+1)+1)
    N = waveform.shape[0]
    for i in range(itera):
        v[i:N-i] = np.minimum(v[i:N-i], (v[:N-2*i] + v[2*i:])/2)
    w = np.power(np.exp(np.exp(v) - 1) - 1, 2) - 1 + wm
    return w

def vali_base(waveform, m_l, thres):
    m = np.median(waveform[waveform < np.median(waveform)])
    vali = np.where(waveform - m > thres, 1, 0) # valid waveform, not dark noise
    pos = omi2pos(vali)
    pos = rm_frag(pos, m_l)
    vali = pos2omi(pos, waveform.shape[0])
    return vali

def deduct_base(waveform, m_l=None, thres=None, itera=20, mode='fast'):
    wave = waveform - np.min(waveform)
    baseline = snip_baseline(wave, itera)
    wave = wave - baseline
    if mode == 'detail':
        wave = wave - find_base(wave, m_l, thres)
    elif mode == 'fast':
        wave = wave - find_base_fast(wave)
    return wave

def find_base(waveform, m_l, thres):
    vali = vali_base(waveform, m_l, thres)
    base_line = np.mean(waveform[vali == 0])
    return base_line

def find_base_fast(waveform):
    m = np.median(waveform[waveform < np.median(waveform)])
    base_line = np.mean(waveform[np.logical_and(waveform < m + 4, waveform > m - 4)])
    return base_line

def omi2pos(vali):
    vali_t = np.concatenate((np.array([0]), vali, np.array([0])), axis=0)
    dval = np.diff(vali_t)
    pos_begin = np.argwhere(dval == 1).flatten()
    pos_end = np.argwhere(dval == -1).flatten()
    pos = np.concatenate((pos_begin.reshape(pos_begin.shape[0], 1), pos_end.reshape(pos_end.shape[0], 1)), axis = 1).astype(np.int16)
    return pos

def pos2omi(pos, len_n):
    vali = np.zeros(len_n).astype(np.int16)
    for i in range(pos.shape[0]):
        vali[pos[i][0]:pos[i][1]] = 1
    return vali

def rm_frag(pos, m_l):
    n = pos.shape[0]
    pos_t = []
    for i in range(n):
        if pos[i][1] - pos[i][0] > m_l:
            pos_t.append(pos[i])
    pos = np.array(pos_t)
    return pos

def pf_to_tw(pf, thres=0.1):
    assert thres < 1, 'thres is too large, which is {}'.format(thres)
    if np.max(pf) < thres:
        t = np.argmax(pf)
        pf = np.zeros_like(pf)
        pf[t] = 1
    pwe = pf[pf > thres]
    pwe = pwe
    pet = np.argwhere(pf > thres).flatten()
    return pet, pwe