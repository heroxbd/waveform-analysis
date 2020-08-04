# -*- coding: utf-8 -*-

import os
import math
import numpy as np
np.set_printoptions(suppress=True)
import scipy
import scipy.stats
from scipy.fftpack import fft, ifft
from scipy import optimize as opti
from scipy.signal import convolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_axes_aligner import align
import numba
from JPwaptool import JPwaptool
import h5py
from scipy.interpolate import interp1d

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['lines.markersize'] = 4.0
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['mathtext.fontset'] = 'cm'

def xiaopeip(wave, spe_pre, eta=0):
    l = len(wave)
    flag = 1
    lowp = np.argwhere(wave > spe_pre['thres']).flatten()
#     lowp = rm_frag(lowp)
    if len(lowp) != 0:
        fitp = np.arange(lowp.min() - spe_pre['mar_l'], lowp.max() + spe_pre['mar_r'])
        fitp = np.unique(np.clip(fitp, 0, len(wave)-1))
        pet = lowp - spe_pre['peak_c']
        pet = np.unique(np.clip(pet, 0, len(wave)-1))
        if len(pet) != 0:
#             pwe, ped = xiaopeip_core(wave, spe_pre['spe'], fitp, pet, eta=eta)
            pwe = xiaopeip_core(wave, spe_pre['spe'], fitp, pet, eta=eta)
        else:
            flag = 0
    else:
        flag = 0
    if flag == 0:
        pet = np.array([np.argmax(wave[spe_pre['peak_c']:])])
        pwe = np.array([1])
#     return pet, pwe, ped
    return pet, pwe

# def xiaopeip_core(wave, spe, fitp, possible, eta=0):
#     l = len(wave)
#     spe = np.concatenate([spe, np.zeros(l - spe.shape[0])])
#     ans0 = np.zeros(len(possible)+1).astype(np.float64)
#     ans0[-1] = wave.min()
#     b = np.zeros((len(possible)+1, 2)).astype(np.float64)
#     b[-1, 0] = -np.inf
#     b[:, 1] = np.inf
#     mne = spe[np.mod(fitp.reshape(fitp.shape[0], 1) - possible.reshape(1, possible.shape[0]), l)]
#     ans = opti.fmin_l_bfgs_b(norm_fit, ans0, args=(mne, wave[fitp], eta), approx_grad=True, bounds=b, maxfun=500000)
#     # ans = opti.fmin_slsqp(norm_fit, ans0, args=(mne, wave[fitp]), bounds=b, iprint=-1, iter=500000)
#     # ans = opti.fmin_tnc(norm_fit, ans0, args=(mne, wave[fitp]), approx_grad=True, bounds=b, messages=0, maxfun=500000)
#     pf = ans[0]
#     return pf[:-1], pf[-1]

# def norm_fit(x, M, y, eta=0):
#     return np.power(y - x[-1] - np.matmul(M, x[:-1]), 2).sum() + eta * x.sum()

def xiaopeip_core(wave, spe, fitp, possible, eta=0):
    l = len(wave)
    spe = np.concatenate([spe, np.zeros(l - spe.shape[0])])
    ans0 = np.zeros(len(possible)).astype(np.float64)
    b = np.zeros((len(possible), 2)).astype(np.float64)
    b[:, 1] = np.inf
    mne = spe[np.mod(fitp.reshape(fitp.shape[0], 1) - possible.reshape(1, possible.shape[0]), l)]
    try:
        ans = opti.fmin_l_bfgs_b(norm_fit, ans0, args=(mne, wave[fitp], eta), approx_grad=True, bounds=b, maxfun=500000)
    except ValueError:
        ans = [np.ones(len(possible)) * 0.2]
    # ans = opti.fmin_slsqp(norm_fit, ans0, args=(mne, wave[fitp]), bounds=b, iprint=-1, iter=500000)
    # ans = opti.fmin_tnc(norm_fit, ans0, args=(mne, wave[fitp]), approx_grad=True, bounds=b, messages=0, maxfun=500000)
    return ans[0]

def norm_fit(x, M, y, eta=0):
    return np.power(y - np.matmul(M, x), 2).sum() + eta * x.sum()

def rm_frag(lowp):
    t = np.argwhere(np.diff(lowp) > 1).flatten()
    ind = np.vstack((np.insert(t + 1, 0, 0), np.append(t, len(lowp)))).T
    slices = [lowp[ind[i][0] : ind[i][1]] for i in range(len(ind))]
    t = [slices[i] for i in range(len(slices)) if len(slices[i]) > 1]
    if len(t) == 0:
        lowp = np.array([])
    else:
        lowp = np.concatenate((t), axis=0)
    return lowp

def lucyddm(waveform, spe_pre, iterations=100):
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
    moveDelta = 9
    spe = np.append(np.zeros(len(spe_pre['spe']) - 2 * moveDelta - 1), np.abs(spe_pre['spe']))
    waveform = np.where(waveform < 0, 0.0001, waveform)
    waveform = waveform.astype(np.float)
    spe = spe.astype(np.float)
    waveform = waveform / np.sum(spe)
    wave_deconv = np.array(waveform)
    spe_mirror = spe[::-1]
    for _ in range(iterations):
        relative_blur = waveform / convolve(wave_deconv, spe, mode='same')
        wave_deconv *= convolve(relative_blur, spe_mirror, mode='same')
        # there is no need to set the bound if the spe and the wave are all none negative 
    return np.arange(0, len(waveform)-moveDelta), wave_deconv[moveDelta:]

def waveformfft(wave, spe_pre):
    length = len(wave)
    spefft = fft(spe_pre['spe'], 2*length)
    wavef = fft(wave, 2*length)
    wavef[(length-int(length*0.7)):(length+int(length*0.7))] = 0
    signalf = np.true_divide(wavef, spefft)
    recon = np.real(ifft(signalf, 2*length))
    return np.arange(length), recon[:length]

def threshold(wave, spe_pre):
    pet = np.argwhere(wave[spe_pre['peak_c']:] > spe_pre['thres'] * 2).flatten()
    pwe = wave[spe_pre['peak_c']:][pet]
    pwe = pwe / pwe.sum() * np.abs(wave.sum()) / spe_pre['spe'].sum()
    if len(pet) == 0:
        pet = np.array([np.argmax(wave[spe_pre['peak_c']:])])
        pwe = np.array([1])
    return pet, pwe

def read_model(spe_path):
    with h5py.File(spe_path, 'r', libver='latest', swmr=True) as speFile:
        cid = speFile['SinglePE'].attrs['ChannelID']
        epulse = speFile['SinglePE'].attrs['Epulse']
        spe = speFile['SinglePE'].attrs['SpePositive']
        thres = speFile['SinglePE'].attrs['Thres']
        spe_pre = {}
        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(111)
        for i in range(len(spe)):
            peak_c = np.argmax(spe[i]); t = np.argwhere(spe[i][peak_c:] < 0.1).flatten()[0] + peak_c
            mar_l = np.sum(spe[i][:peak_c] < thres[i])
            mar_r = np.sum(spe[i][peak_c:t] < thres[i])
            spe_pre_i = {'spe':spe[i], 'epulse':epulse, 'peak_c':peak_c, 'mar_l':mar_l, 'mar_r':mar_r, 'thres':thres[i]}
            spe_pre.update({cid[i]:spe_pre_i})
            ax.plot(spe_pre[cid[i]]['spe'])
        ax.grid()
        ax.set_xlabel('$Time/\mathrm{ns}$')
        ax.set_ylabel('$Voltage/\mathrm{mV}$')
        fig.savefig('img/spe.png', bbox_inches='tight')
        plt.close()
    return spe_pre

def clip(pet, pwe, thres):
    if len(pet[pwe > thres]) == 0:
        pet = np.array([pet[np.argmax(pwe)]])
        pwe = np.array([1])
    else:
        pet = pet[pwe > thres]
        pwe = pwe[pwe > thres]
    return pet, pwe

def snip_baseline(waveform, itera=20):
    wm = np.min(waveform)
    waveform = waveform - wm
    v = np.log(np.log(np.sqrt(waveform+1)+1)+1)
    N = waveform.shape[0]
    for i in range(itera):
        v[i:N-i] = np.minimum(v[i:N-i], (v[:N-2*i] + v[2*i:])/2)
    w = np.power(np.exp(np.exp(v) - 1) - 1, 2) - 1 + wm
    return w

def demo(pet, pwe, tth, spe_pre, leng, wave, cid, mode, full=False):
    penum = len(tth)
    print('PEnum is {}'.format(penum))
    pf0 = np.zeros(leng); pf1 = np.zeros(leng)
    if mode == 'Weight':
        tru_pet = tth['RiseTime']
        t, c = np.unique(tru_pet, return_counts=True)
        pf0[t] = c
        pf1[pet] = pwe
        xlabel = '$PEnum/\mathrm{1}$'
        distd = '(W/ns,P/1)'; distl = 'pdist'
        Q = penum; q = np.sum(pwe)
        edist = np.abs(Q - q) * scipy.stats.poisson.pmf(Q, Q)
    elif mode == 'Charge':
        t = tth['RiseTime']; w = tth[mode]
        t = np.unique(t)
        c = np.array([np.sum(w[tth['RiseTime'] == i]) for i in t])
        pf0[t] = c / spe_pre['spe'].sum()
        pf1[pet] = pwe / spe_pre['spe'].sum()
        xlabel = '$Charge/\mathrm{mV}\cdot\mathrm{ns}$'
        distd = '(W/ns,C/mV*ns)'; distl = 'cdiff'
        edist = pwe.sum() - c.sum()
    print('truth RiseTime = {}, Weight = {}'.format(t, c))
    wave0 = np.convolve(spe_pre['spe'], pf0, 'full')[:leng]
    print('truth Resi-norm = {}'.format(np.linalg.norm(wave-wave0)))
    print('RiseTime = {}, Weight = {}'.format(pet, pwe))
    wdist = scipy.stats.wasserstein_distance(t, pet, u_weights=c, v_weights=pwe)
    print('wdist = {},'.format(wdist)+distl+' = {}'.format(edist))
    wave1 = np.convolve(spe_pre['spe'], pf1, 'full')[:leng]
    print('Resi-norm = {}'.format(np.linalg.norm(wave-wave1)))

    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot(111)
    ax.grid()
    ax2 = ax.twinx()
    ax.plot(wave, c='b', label='origin wave')
    ax.plot(wave0, c='k', label='truth wave')
    ax.plot(wave1, c='C1', label='recon wave')
    ax.set_xlabel('$Time/\mathrm{ns}$')
    ax.set_ylabel('$Voltage/\mathrm{mV}$')
    ax.hlines(spe_pre['thres'], 0, 1029, color='c', label='threshold')
    ax2.set_ylabel(xlabel)
    fig.suptitle('eid={},cid={},'.format(tth['EventID'][0], tth['ChannelID'][0])+distd+'-dist={:.2f},{:.2f}'.format(wdist, edist))
    ax2.vlines(t, 0, c, color='g', label='truth '+mode)
    ax2.vlines(pet, -pwe, 0, color='y', label='recon '+mode)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    align.yaxes(ax, 0, ax2, 0)
    ax2.legend(lines + lines2, labels + labels2)
    if full:
        ax.set_xlim(max(t.min()-50, 0), min(t.max()+150, leng))
    fig.savefig('img/demoe{}c{}.png'.format(tth['EventID'][0], tth['ChannelID'][0]), bbox_inches='tight')
    fig.clf()
    plt.close(fig)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(spe_pre['spe'], c='b')
    ax.grid()
    ax.set_xlabel('$Time/\mathrm{ns}$')
    ax.set_ylabel('$Voltage/\mathrm{mV}$')
    fig.savefig('img/spe{}.png'.format(cid), bbox_inches='tight')
    fig.clf()
    return
