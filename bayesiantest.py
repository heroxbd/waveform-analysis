import sys
sys.path.insert(0,'')
import h5py
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.rcParams['font.size'] = 15
import matplotlib.pyplot as plt
import argparse
import mcp
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import wf_func as wff
from scipy import integrate

import itertools
print(mcp.version)
window = wff.window
gmu = wff.gmu
gsigma = wff.gsigma
p = wff.p
p[2] = p[2] * gmu / integrate.quad(lambda t : wff.spe(np.array([t]), tau=p[0], sigma=p[1], A=p[2]), 0, 100)[0]
std = wff.std
def genwave(pelist, window, noise=False):
    if noise:
        wave = rng.normal(0, baselinerms, window)
    else:
        wave = np.zeros(window)
    # wave += baseline
    pan = np.arange(0, window, 1 / wff.nshannon)
    for i in range(pelist.shape[0]):
        # charge * charge2Amp = the times need to multiply waveform
        wave += wff.spe(((pan-pelist[i]['HitPosInWindow'])+np.abs(pan-pelist[i]['HitPosInWindow']))/2, tau=p[0], sigma=p[1], A=p[2])*pelist[i]['Charge']/gmu
    return wave
psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('-n', type=int, help='bin split number', default=1)
psr.add_argument('-R', type=int, help='whether use real prior distribution', default=0)
psr.add_argument('-b',dest='base',default=0,help='whether remove the baseline')
psr.add_argument('--ref', type=str, help='reference file')
psr.add_argument('-p',dest='progress',default=False, action='store_true')
psr.add_argument('--prior', default=False, action='store_true')
args = psr.parse_args()
n = args.n
prior = args.prior
spe_pre = wff.read_model(args.ref, 1)

with h5py.File(args.ipt, 'r') as ipt:
    waveforms = ipt['Readout/Waveform'][:]
    pelist = ipt['SimTriggerInfo/PEList'][:]
    t0_truth = ipt['SimTruth/T'][:]
    # mu = ipt.attrs['mu']
    Mu = ipt['Readout/Waveform'].attrs['mu'].item()
    Tau = ipt['Readout/Waveform'].attrs['tau'].item()
    Sigma = ipt['Readout/Waveform'].attrs['sigma'].item()
    gmu = ipt['SimTriggerInfo/PEList'].attrs['gmu'].item()
    gsigma = ipt['SimTriggerInfo/PEList'].attrs['gsigma'].item()
    s0 = spe_pre[0]['std'] / np.linalg.norm(spe_pre[0]['spe'])
N = waveforms.shape[0]
window = waveforms[0]['Waveform'].shape[0]
testN = 101
# print('test dataset choose the front {} wave from {} wave'.format(testN,triggers.shape))
maxSpe = np.max(spe_pre[0]['spe'])
lightCurve = np.clip(wff.convolve_exp_norm(np.arange(100), Tau, Sigma), np.finfo(np.float64).tiny, np.inf)
maxQratio = np.max(lightCurve)/np.sum(lightCurve)
Thres = {'lucyddm':0.1, 'fbmp':1e-6}
with PdfPages(args.opt) as opt:
    pedf = pd.DataFrame(pelist).groupby('TriggerNo')
    for i,pes in pedf:
        # pes = pedf.get_group(i)
        wave = waveforms[i]['Waveform']
        cid = waveforms[i]['ChannelID']
        truth = pelist[pelist['TriggerNo'] == waveforms[i]['TriggerNo']]

        # initialTest
        A, wave_r, tlist, t0_t, t0_delta, cha, left_wave, right_wave = wff.initial_params(wave[::wff.nshannon], spe_pre[cid], Tau, Sigma, gmu, Thres['lucyddm'], p, is_t0=True, is_delta=False, n=n, nshannon=1)
        fig, ax = plt.subplots(dpi=100,figsize=(16,8))
        ax.plot(np.arange(left_wave,right_wave),wave_r,label='origin wave', alpha=0.5)
        ax.scatter(tlist, cha/gmu*10, label='Lucyddm', s=10)

        factor = np.sqrt(np.diag(np.matmul(A.T, A)))
        A = A/factor
        # get the prior estimate
        mu_t = abs(wave.sum() / gmu)
        maxQ = int(mu_t*maxQratio+np.sqrt(mu_t))
        if args.R == 1:
            la = np.sum(pelist['TriggerNo']==i) * wff.convolve_exp_norm(tlist - t0_truth[i]['T0'],Tau,Sigma)/n +1e-8
        else:
            la = mu_t * wff.convolve_exp_norm(tlist - t0_truth[i]['T0'],Tau,Sigma)/n+1e-8 
        ax.plot(tlist, la*(maxSpe/np.max(lightCurve)), label='expect light curve')

        # pes = pelist[pelist['TriggerNo']==i]
        pest = pes['HitPosInWindow']
        pesc = pes['Charge']
        ax.scatter(pest,pesc/gmu*10, alpha=0.5,s=50,label='pe truth')
        truthwave = genwave(pes.to_records(), window,noise=False)
        ax.plot(truthwave,label='truth wave')

        expectx, expectx_star, psy_star, nu_star,nu_star_bk, T_star, d_max_i, num_i = wff.fbmpr_fxn_reduced(wave_r, A, spe_pre[cid]['std'] ** 2, (gsigma * factor / gmu) ** 2, factor, len(la), p1=la, stop=5, truth=truth, i=i, left=left_wave, right=right_wave, tlist=tlist, gmu=gmu, para=p, prior=prior)
        c_star = np.zeros_like(expectx_star).astype(int)
        for k in range(len(T_star)):
            t, c = np.unique(T_star[k], return_counts=True)
            c_star[k, t] = c
        xmmse_star = c_star
        ax.scatter(tlist[xmmse_star[0]>0], np.array((expectx_star[0]/factor)[xmmse_star[0]>0])*10, s=20, label='fbmp')
        fbmpwave = genwave(np.rec.fromarrays([tlist[xmmse_star[0]>0], np.array((expectx_star[0]/factor*gmu)[xmmse_star[0]>0])],dtype=[('HitPosInWindow',np.float64),('Charge',np.float64)]),500,noise=False)
        # assert False
        ax.plot(fbmpwave, label='fbmp wave')

        ax.set_title('TriggerNo {}, truth:{}, fbmp:{}'.format(i, pes.shape[0],np.sum(xmmse_star[0])))
        ax.set_xlim([left_wave, right_wave])
        ax.legend()
        opt.savefig(figure=fig)
        print('eid',i)
        print(list(zip(tlist[xmmse_star[0]>0],np.array((expectx_star[0]/factor*gmu)[xmmse_star[0]>0]))))
        print(list(zip(pes['HitPosInWindow'].values,pes['Charge'].values)))
        if i>testN:
            break
