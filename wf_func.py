# -*- coding: utf-8 -*-

import os
import math

import numpy as np
np.set_printoptions(suppress=True)
import scipy
import scipy.stats
from scipy.stats import poisson, uniform, norm
from scipy.fftpack import fft, ifft
from scipy import optimize as opti
import scipy.special as special
from scipy.signal import convolve
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_axes_aligner import align
import h5py
from scipy.interpolate import interp1d
import numba

plt.style.use('classic')
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 20
plt.rcParams['lines.markersize'] = 4.0
plt.rcParams['lines.linewidth'] = 2.0
# plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = True
plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams['axes.unicode_minus'] = False

def xiaopeip(wave, spe_pre, eta=0):
    l = len(wave)
    flag = 1
    lowp = np.argwhere(wave > 5 * spe_pre['std']).flatten()
#     lowp = rm_frag(lowp)
    if len(lowp) != 0:
        fitp = np.arange(lowp.min() - spe_pre['mar_l'], lowp.max() + spe_pre['mar_r'])
        fitp = np.unique(np.clip(fitp, 0, len(wave)-1))
        pet = lowp - spe_pre['peak_c']
        pet = np.unique(np.clip(pet, 0, len(wave)-1))
        if len(pet) != 0:
#             pwe, ped = xiaopeip_core(wave, spe_pre['spe'], fitp, pet, eta=eta)
            pwe = xiaopeip_core(wave, spe_pre['spe'], fitp, pet, eta=eta)
            pwe = pwe / np.sum(pwe) * np.sum(wave) / np.sum(spe_pre['spe'])
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
#     spe = np.concatenate([spe, np.zeros(l - len(spe))])
#     ans0 = np.zeros(len(possible)+1).astype(np.float64)
#     ans0[-1] = wave.min()
#     b = np.zeros((len(possible)+1, 2)).astype(np.float64)
#     b[-1, 0] = -np.inf
#     b[:, 1] = np.inf
#     mne = spe[np.mod(fitp.reshape(len(fitp), 1) - possible.reshape(1, len(possible)), l)]
#     ans = opti.fmin_l_bfgs_b(norm_fit, ans0, args=(mne, wave[fitp], eta), approx_grad=True, bounds=b, maxfun=500000)
#     # ans = opti.fmin_slsqp(norm_fit, ans0, args=(mne, wave[fitp]), bounds=b, iprint=-1, iter=500000)
#     # ans = opti.fmin_tnc(norm_fit, ans0, args=(mne, wave[fitp]), approx_grad=True, bounds=b, messages=0, maxfun=500000)
#     pf = ans[0]
#     return pf[:-1], pf[-1]

# def norm_fit(x, M, y, eta=0):
#     return np.power(y - x[-1] - np.matmul(M, x[:-1]), 2).sum() + eta * x.sum()

def xiaopeip_core(wave, spe, fitp, possible, eta=0):
    l = len(wave)
    spe = np.concatenate([spe, np.zeros(l - len(spe))])
    ans0 = np.ones(len(possible)).astype(np.float64)
    b = np.zeros((len(possible), 2)).astype(np.float64)
    b[:, 1] = np.inf
    mne = spe[np.mod(fitp.reshape(len(fitp), 1) - possible.reshape(1, len(possible)), l)]
    try:
        ans = opti.fmin_l_bfgs_b(norm_fit, ans0, args=(mne, wave[fitp], eta), approx_grad=True, bounds=b, maxfun=500000)
#         ans = opti.fmin_l_bfgs_b(wdist_fit, ans0, args=(mne, wave[fitp], eta), approx_grad=True, bounds=b, maxfun=500000)
    except ValueError:
        ans = [np.ones(len(possible)) * 0.2]
    # ans = opti.fmin_slsqp(norm_fit, ans0, args=(mne, wave[fitp]), bounds=b, iprint=-1, iter=500000)
    # ans = opti.fmin_tnc(norm_fit, ans0, args=(mne, wave[fitp]), approx_grad=True, bounds=b, messages=0, maxfun=500000)
    return ans[0]

def norm_fit(x, M, y, eta=0):
    return np.power(y - np.matmul(M, x), 2).sum() + eta * x.sum()

def wdist_fit(x, M, y, eta=0):
    r = np.matmul(M, x)
    return np.sum(np.abs(np.cumsum(r) / np.sum(r) - np.cumsum(y) / np.sum(y)))

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
    spe = np.append(np.zeros(len(spe_pre) - 2 * 9 - 1), np.abs(spe_pre))
    waveform = np.where(waveform <= 0, 1e-6, waveform)
    spe = np.where(spe <= 0, 1e-6, spe)
    waveform = waveform / np.sum(spe)
    wave_deconv = waveform.copy()
    spe_mirror = spe[::-1]
    for _ in range(iterations):
        relative_blur = waveform / np.convolve(wave_deconv, spe, mode='same')
        wave_deconv *= np.convolve(relative_blur, spe_mirror, mode='same')
    return np.arange(0, len(waveform) - 9), wave_deconv[9:]

def waveformfft(wave, spe_pre):
    length = len(wave)
    spefft = fft(spe_pre['spe'], 2*length)
    wavef = fft(wave, 2*length)
    wavef[(length-int(length*0.8)):(length+int(length*0.8))] = 0
    signalf = np.true_divide(wavef, spefft)
    recon = np.real(ifft(signalf, 2*length))
    return np.arange(length), recon[:length]

def threshold(wave, spe_pre):
    pet = np.argwhere(wave[spe_pre['peak_c']:] > spe_pre['std'] * 5).flatten()
    pwe = wave[spe_pre['peak_c']:][pet]
    pwe = pwe / pwe.sum() * np.abs(wave.sum()) / spe_pre['spe'].sum()
    if len(pet) == 0:
        pet = np.array([np.argmax(wave[spe_pre['peak_c']:])])
        pwe = np.array([1])
    return pet, pwe

def findpeak(wave, spe_pre):
    w = savgol_filter(wave, 11, 2)
    dpta = np.where(np.diff(w, prepend=w[0]) > 0, 1, -1)
    dpta = np.diff(dpta, prepend=dpta[0])
    petr = np.argwhere((w > spe_pre['std'] * 5) & (dpta < 0)).flatten() - spe_pre['peak_c']
    pet = petr[petr >= 0]
    pwe = wave[pet + spe_pre['peak_c']]
    pwe = pwe / np.sum(pwe) * np.abs(np.sum(wave)) / np.sum(spe_pre['spe'])
    if len(pet) == 0:
        pet = np.array([np.argmax(wave[spe_pre['peak_c']:])])
        pwe = np.array([1])
    return pet, pwe

def read_model(spe_path):
    with h5py.File(spe_path, 'r', libver='latest', swmr=True) as speFile:
        cid = speFile['SinglePE'].attrs['ChannelID']
        epulse = speFile['SinglePE'].attrs['Epulse']
        spe = speFile['SinglePE'].attrs['SpePositive']
        std = speFile['SinglePE'].attrs['Std']
        if 'parameters' in list(speFile['SinglePE'].attrs.keys()):
            p = speFile['SinglePE'].attrs['parameters']
        else:
            p = [None,] * len(spe)
        spe_pre = {}
        fig = plt.figure()
        # fig.tight_layout()
        gs = gridspec.GridSpec(1, 1, figure=fig, left=0.1, right=0.85, top=0.95, bottom=0.15, wspace=0.4, hspace=0.5)
        ax = fig.add_subplot(gs[0, 0])
        for i in range(len(spe)):
            peak_c = np.argmax(spe[i]); t = np.argwhere(spe[i][peak_c:] < 0.1).flatten()[0] + peak_c
            mar_l = np.sum(spe[i][:peak_c] < 5 * std[i])
            mar_r = np.sum(spe[i][peak_c:t] < 5 * std[i])
            spe_pre_i = {'spe':spe[i], 'epulse':epulse, 'peak_c':peak_c, 'mar_l':mar_l, 'mar_r':mar_r, 'std':std[i], 'parameters':p[i]}
            spe_pre.update({cid[i]:spe_pre_i})
            ax.plot(spe_pre[cid[i]]['spe'])
        ax.grid()
        ax.set_xlabel(r'$Time/\mathrm{ns}$')
        ax.set_ylabel(r'$Voltage/\mathrm{mV}$')
        fig.savefig('Note/figures/pmtspe.pgf')
        fig.savefig('Note/figures/pmtspe.pdf')
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

def glow(n, tau):
    return np.random.exponential(tau, size=n)

def transit(n, sigma):
    return np.random.normal(0, sigma, size=n)

def time(n, tau, sigma):
    if tau == 0:
        return np.sort(transit(n, sigma))
    elif sigma == 0:
        return np.sort(glow(n, tau))
    else:
        return np.sort(glow(n, tau) + transit(n, sigma))

def convolve_exp_norm(x, tau, sigma):
    if tau == 0.:
        y = norm.pdf(x, loc=0, scale=sigma)
    elif sigma == 0.:
        y = np.where(x >= 0., 1/tau * np.exp(-x/tau), 0.)
    else:
        alpha = 1/tau
        co = alpha/2. * np.exp(alpha*alpha*sigma*sigma/2.)
        x_erf = (alpha*sigma*sigma - x)/(np.sqrt(2.)*sigma)
        y = co * (1. - special.erf(x_erf)) * np.exp(-alpha*x)
    return y

def spe(t, tau, sigma, A):
    s = np.zeros_like(t).astype(np.float)
    t0 = t[t > np.finfo(np.float64).tiny]
    s[t > np.finfo(np.float64).tiny] = A * np.exp(-1 / 2 * (np.log(t0 / tau) * np.log(t0 / tau) / sigma / sigma))
    return s

def charge(n, gmu, gsigma=40):
    gsigma = 40.
    chargesam = norm.ppf(1 - uniform.rvs(scale=1-norm.cdf(0, loc=gmu, scale=gsigma), size=n), loc=gmu, scale=gsigma)
    return chargesam

def lasso_select(pf_r, wave, mne, E):
    pf_r = pf_r[np.argmin([loss(pf_r[j], mne, wave, E) for j in range(len(pf_r))])]
    return pf_r

def loss(x, M, y, eta=0.):
    return np.power(y - np.matmul(M, x), 2).sum() + eta * x.sum()

def probcharge(charge, n, gmu, gsigma=40):
    gmu = gmu * n
    gsigma = gsigma * np.sqrt(n)
    prob = norm.pdf(charge, loc=gmu, scale=gsigma) / (1 - norm.cdf(0, loc=gmu, scale=gsigma))
    return prob

def demo(pet, pwe, tth, spe_pre, window, wave, cid, full=False, fold='Note/figures', ext='.pgf'):
    penum = len(tth)
    print('PEnum is {}'.format(penum))
    pf0 = np.zeros(window)
    pf1 = np.zeros(window)
    t = tth['HitPosInWindow']
    w = tth['Charge']
    tu = np.unique(t)
    cu = np.array([np.sum(w[tth['HitPosInWindow'] == i]) for i in tu])
    tuint = np.around(tu).astype(np.int)
    for i in tu:
        pf0[i] = pf0[i] + cu[i]
    pf0 = pf0 / spe_pre['spe'].sum()
    pf1[pet] = pwe / spe_pre['spe'].sum()
    ylabel = '$Charge/\mathrm{mV}\cdot\mathrm{ns}$'
    distd = '(W/ns,C/mV*ns)'; distl = 'cdiff'
    edist = pwe.sum() - w.sum()
    print('truth HitPosInWindow = {}, Weight = {}'.format(t, w))
    wave0 = np.convolve(spe_pre['spe'], pf0, 'full')[:window]
    print('truth Resi-norm = {}'.format(np.linalg.norm(wave-wave0)))
    print('HitPosInWindow = {}, Weight = {}'.format(pet, pwe))
    wdist = scipy.stats.wasserstein_distance(t, pet, u_weights=w, v_weights=pwe)
    print('wdist = {}, '.format(wdist)+distl+' = {}'.format(edist))
    wave1 = np.convolve(spe_pre['spe'], pf1, 'full')[:window]
    print('Resi-norm = {}'.format(np.linalg.norm(wave-wave1)))

    fig = plt.figure(figsize=(10, 10))
    fig.tight_layout()
    ax0 = fig.add_axes((.1, .2, .85, .3))
    ax0.plot(wave, c='b', label='origin wave')
    ax0.plot(wave0, c='k', label='truth wave')
    ax0.plot(wave1, c='g', label='recon wave')
    ax0.set_ylabel('$Voltage/\mathrm{mV}$')
    ax0.hlines(5 * spe_pre['std'], 0, 1029, color='c', label='threshold')
    ax0.set_xticklabels([])
    ax0.set_ylim(min(wave)-5, max(wave)+5)
    ax0.legend(loc=1)
    ax0.grid()
    if full:
        ax0.set_xlim(0, window)
    else:
        ax0.set_xlim(max(t.min()-50, 0), min(t.max()+150, window))
    ax1 = fig.add_axes((.1, .5, .85, .2))
    ax1.vlines(tu, 0, cu, color='k', label='truth Charge')
    ax1.set_ylabel(ylabel)
    ax1.set_xticklabels([])
    ax1.set_xlim(ax0.get_xlim())
    ax1.set_ylim(0, max(max(cu), max(pwe))*1.1)
    ax1.set_yticks(np.arange(0, max(max(cu), max(pwe)), 50))
    ax1.legend(loc=1)
    ax1.grid()
    ax2 = fig.add_axes((.1, .7, .85, .2))
    ax2.vlines(pet, 0, pwe, color='g', label='recon Charge')
    ax2.set_ylabel(ylabel)
    ax2.set_xticklabels([])
    ax2.set_xlim(ax0.get_xlim())
    ax2.set_ylim(0, max(max(cu), max(pwe))*1.1)
    ax2.set_yticks(np.arange(0, max(max(cu), max(pwe)), 50))
    ax2.legend(loc=1)
    ax2.grid()
    ax3 = fig.add_axes((.1, .1, .85, .1))
    ax3.scatter(np.arange(window), wave1 - wave, c='k', label='residual wave', marker='.')
    ax3.set_xlabel('$t/\mathrm{ns}$')
    ax3.set_ylabel('$Voltage/\mathrm{mV}$')
    ax3.set_xlim(ax0.get_xlim())
    dh = int((max(np.abs(wave1 - wave))//5+1)*5)
    ax3.set_yticks(np.linspace(-dh, dh, int(2*dh//5+1)))
    ax3.legend(loc=1)
    ax3.grid()
    if ext != '.pgf':
        fig.suptitle('eid={},cid={},'.format(tth['TriggerNo'][0], tth['PMTId'][0])+distd+'-dist={:.2f},{:.2f}'.format(wdist, edist), y=0.95)
    fig.savefig(fold + '/demoe{}c{}'.format(tth['TriggerNo'][0], tth['PMTId'][0]) + ext)
    fig.savefig(fold + '/demoe{}c{}'.format(tth['TriggerNo'][0], tth['PMTId'][0]) + '.pdf')
    fig.clf()
    plt.close(fig)

    fig = plt.figure()
    # fig.tight_layout()
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.1, right=0.85, top=0.95, bottom=0.15, wspace=0.4, hspace=0.5)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(spe_pre['spe'], c='b')
    ax.grid()
    ax.set_xlabel('$t/\mathrm{ns}$')
    ax.set_ylabel('$Voltage/\mathrm{mV}$')
    fig.savefig(fold + '/spe{:02d}'.format(cid) + ext)
    fig.savefig(fold + '/spe{:02d}'.format(cid) + '.pdf')
    fig.clf()
    plt.close(fig)
    return