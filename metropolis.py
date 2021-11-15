import time
import argparse

import h5py
import numpy as np
# np.seterr(all='raise')
from scipy.stats import poisson

import wf_func as wff
from functools import partial

global_start = time.time()
cpu_global_start = time.process_time()

psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('ipt', type=str, help='input file')
psr.add_argument('--ref', type=str, help='reference file')
args = psr.parse_args()

fipt = args.ipt
fopt = args.opt
reference = args.ref

spe_pre = wff.read_model(reference, 1)
with h5py.File(fipt, 'r', libver='latest', swmr=True) as ipt:
    ent = ipt['Readout/Waveform'][:]
    pelist = ipt['SimTriggerInfo/PEList'][:]
    t0_truth = ipt['SimTruth/T'][:]
    N = len(ent)
    print('{} waveforms will be computed'.format(N))
    window = len(ent[0]['Waveform'])
    assert window >= len(spe_pre[0]['spe']), 'Single PE too long which is {}'.format(len(spe_pre[0]['spe']))
    Mu = ipt['Readout/Waveform'].attrs['mu'].item()
    Tau = ipt['Readout/Waveform'].attrs['tau'].item()
    Sigma = ipt['Readout/Waveform'].attrs['sigma'].item()
    gmu = ipt['SimTriggerInfo/PEList'].attrs['gmu'].item()
    gsigma = ipt['SimTriggerInfo/PEList'].attrs['gsigma'].item()
    PEList = ipt['SimTriggerInfo/PEList'][:]

p = spe_pre[0]['parameters']
if Tau != 0:
    Alpha = 1 / Tau
    Co = (Alpha / 2. * np.exp(Alpha ** 2 * Sigma ** 2 / 2.)).item()
std = 1.
Thres = wff.Thres
mix0sigma = 1e-3
mu0 = np.arange(1, int(Mu + 5 * np.sqrt(Mu)))
n_t = np.arange(1, 20)

n = 1
b_t0 = [0., 600.]

print('Initialization finished, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - global_start, time.process_time() - cpu_global_start))
tic = time.time()
cpu_tic = time.process_time()

ent = np.sort(ent, kind='stable', order=['TriggerNo', 'ChannelID'])
Chnum = len(np.unique(ent['ChannelID']))

dt = [('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('flip', np.int8), ('delta_nu', np.float64)]
mu0_dt = [('TriggerNo', np.uint32), ('ChannelID', np.uint32), ('mu_t', np.float64)]

d_history = [('TriggerNo', np.uint32), ('ChannelID', np.uint32), 
             ('step', np.uint32), ('loc', np.float32)]
TRIALS = 2000

def combine(A, cx, t):
    '''
    combine neighbouring dictionaries to represent sub-bin locations
    '''
    frac, ti = np.modf(t - 0.5)
    ti = int(ti)
    alpha = np.array((1 - frac, frac))
    return alpha @ A[:, ti:(ti+2)].T, alpha @ cx[:, ti:(ti+2)].T

def flow(cx, p1, z, N, sig2s, mus, A, p_cha, mu_t):
    '''
    flow
    ====
    连续时间游走

    s: list of PE locations
    mu_t: LucyDDM 的估算 PE 数
    '''
    # idx: [0, 1) 之间的随机数，用于点中 PE
    istar = np.random.rand(TRIALS)
    # 同时可用于创生位置的选取
    c_cha = np.cumsum(p_cha)
    home_s = np.interp(istar, xp = np.insert(c_cha, 0, 0), fp = np.arange(N+1))

    NPE0 = int(mu_t + 0.5) # 取整
    # t 的位置，取值为 [0, N)
    s = list(np.interp((np.arange(NPE0) + 0.5) / NPE0, 
                       xp = np.insert(c_cha, 0, 0), fp = np.arange(N+1)))
    for t in s:
        Δν, Δcx, Δz = move(*combine(A, cx, t), z, 1, mus, sig2s, A)
        cx += Δcx
        z += Δz

    si = 0
    es_history = np.zeros(TRIALS * (NPE0 + 5) * N, dtype=d_history)

    wander_s = np.random.normal(size=TRIALS)

    # step: +1 创生一个 PE， -1 消灭一个 PE， +2 向左或向右移动
    flip = np.random.choice((-1, 1, 2), TRIALS, p=np.array((7, 7, 2))/16)
    Δν_history = np.zeros(TRIALS) # list of Δν's

    for i, (t, step, home, wander, accept) in enumerate(zip(istar, flip, home_s, wander_s, 
                                                           np.log(np.random.rand(TRIALS)))):
        # 不设左右边界
        NPE = len(s)
        if NPE == 0:
            step = 1 # 只能创生
            accept += np.log(16/7) # 惩罚
        elif NPE == 1 and step == -1:
            # 1 -> 0: 行动后从 0 脱出的几率大，需要鼓励
            accept -= np.log(16/7)

        if step == 1: # 创生
            if home >= 0.5 and home <= N - 0.5: 
                Δν, Δcx, Δz = move(*combine(A, cx, home), z, 1, mus, sig2s, A)
                Δν += np.log(p1[int(home)]) - np.log(p_cha[int(home)]) - np.log(NPE + 1)
                if Δν >= accept:
                    s.append(home)
            else: # p(w|s) 无定义
                Δν = -np.inf
        else:
            op = int(t * NPE) # 操作的 PE 编号
            loc = s[op] # 待操作 PE 的位置
            Δν, Δcx, Δz = move(*combine(A, cx, loc), z, -1, mus, sig2s, A)
            if step == -1: # 消灭
                Δν -= np.log(p1[int(loc)]) - np.log(p_cha[int(loc)]) - np.log(NPE)

                if Δν >= accept:
                    del s[op]
            elif step == 2: # 移动
                nloc = loc + wander # 待操作 PE 的新位置
                if nloc >= 0.5 and nloc <= N - 0.5: # p(w|s) 无定义
                    Δν1, Δcx1, Δz1 = move(*combine(A, cx + Δcx, nloc), z + Δz, 1, mus, sig2s, A)
                    Δν += Δν1
                    Δν += np.log(p1[int(nloc)]) - np.log(p1[int(loc)])
                    if Δν >= accept:
                        s[op] = nloc
                        Δcx += Δcx1
                        Δz += Δz1
                else: # p(w|s) 无定义
                    Δν = -np.inf

        if Δν >= accept:
            cx += Δcx
            z += Δz
            si1 = si + len(s)
            es_history[si:si1]['step'] = i
            es_history[si:si1]['loc'] = s
            si = si1
        else: # reject proposal
            Δν = 0
            step = 0
        Δν_history[i] = Δν
        flip[i] = step
    return flip, Δν_history, es_history[:si1]

def move(A_vec, c_vec, z, step, mus, sig2s, A):
    '''
    step
    ====
    A_vec: 行向量
    c_vec: 行向量
    1: 在 t 加一个 PE
    -1: 在 t 减一个 PE
    '''
    fsig2s = step * sig2s
    # Eq. (30) sig2s = 1 sigma^2 - 0 sigma^2
    beta_under = (1 + fsig2s * np.dot(A_vec, c_vec))
    beta = fsig2s / beta_under

    # Eq. (31) # sign of mus[t] and sig2s[t] cancels
    Δν = 0.5 * (beta * (z @ c_vec + mus / sig2s) ** 2 - mus ** 2 / fsig2s)
    # sign of space factor in Eq. (31) is reversed.  Because Eq. (82) is in the denominator.
    Δν -= 0.5 * np.log(beta_under) # space
    # accept, prepare for the next
    # Eq. (33) istar is now n_pre.  It crosses n_pre and n, thus is in vector form.
    Δcx = -np.einsum('n,m,mp->np', beta * c_vec, c_vec, A, optimize=True)

    # Eq. (34)
    Δz = -step * A_vec * mus
    return Δν, Δcx, Δz

def metropolis(ent, sample, mu0, d_tlist, s_history):
    i_tlist = 0
    si = 0
    for ie, e in enumerate(ent):
        time_fbmp_start = time.time()
        eid = e["TriggerNo"]
        cid = e['ChannelID']
        assert cid == 0
        PEs = PEList[np.logical_and(PEList["TriggerNo"] == eid, PEList["PMTId"] == cid)]
        NPE_t = len(PEs)

        wave = e['Waveform'].astype(np.float64) * spe_pre[cid]['epulse']

        # initialization
        A, y, tlist, t0_t, t0_delta, cha, left_wave, right_wave = wff.initial_params(wave, spe_pre[e['ChannelID']], Tau, Sigma, gmu, Thres['lucyddm'], p, is_t0=False, is_delta=False, n=n)
        s_cha = np.cumsum(cha)
        # moving average filter of size 2*n+1
        cha = np.pad(s_cha[2*n+1:], (n+1, n), 'edge') - np.pad(s_cha[:-(2*n+1)], (n+1, n), 'edge')
        cha += 1e-8 # for completeness of the random walk.
        o_tlist = i_tlist + len(tlist)
        d_tlist[i_tlist:o_tlist] = list(zip(np.repeat(eid, o_tlist - i_tlist),
                                            np.repeat(cid, o_tlist - i_tlist),
                                            tlist, cha))
        i_tlist = o_tlist
        t0_t = t0_truth['T0'][ie] # override with truth to debug mu
        mu_t = abs(y.sum() / gmu)
        mu0[ie] = (eid, cid, mu_t)
        # Eq. (9) where the columns of A are taken to be unit-norm.
        mus = np.sqrt(np.diag(np.matmul(A.T, A))) # ~39
        # elements of mus must be equal for continuous metropolis to function
        assert np.std(mus) < 1e-6, "mus must be equal"
        mus = mus[0]
        A = A / mus

        '''
        A: basis dictionary
        p1: prior probability for each bin.
        sig2w: variance of white noise.
        sig2s: variance of signal x_i.
        mus: mean of signal x_i.
        TRIALS: number of Metropolis steps.
        '''

        p1 = mu_t * wff.convolve_exp_norm(tlist - t0_t, Tau, Sigma) / n

        sig2w = spe_pre[cid]['std'] ** 2
        sig2s = (mus * (gsigma / gmu)) ** 2 # ~94

        # Only for multi-gaussian with arithmetic sequence of mu and sigma
        # N: number of t bins
        # M: length of the waveform clip
        M, N = A.shape

        # nu: nu for all s_n=0.
        ν = -0.5 * np.linalg.norm(y) ** 2 / sig2w - 0.5 * M * np.log(2 * np.pi)
        ν -= 0.5 * M * np.log(sig2w)
        ν += poisson.logpmf(0, p1).sum()

        # Eq. (29)
        cx = A / sig2w
        # mu = 0 => (y - A * mu -> z)
        z = y

        p_cha = cha/np.sum(cha)

        # Metropolis flow
        flip, Δν_history, es_history = flow(cx, p1, z, N, sig2s, mus, A, p_cha, mu_t)

        sample[ie*TRIALS:(ie+1)*TRIALS] = list(zip(np.repeat(eid, TRIALS), 
                                                   np.repeat(cid, TRIALS), 
                                                   flip, Δν_history))
        so = si + len(es_history)
        es_history['TriggerNo'] = eid
        s_history[si:so]  = es_history
        si = so
    d_tlist.resize((o_tlist,))
    s_history.resize((so,))

with h5py.File(fopt, 'w') as opt:
    sample = opt.create_dataset('sample', shape=(N*TRIALS,), dtype=dt)
    mu0 = opt.create_dataset('mu0', shape=(N,), dtype=mu0_dt)
    d_tlist = opt.create_dataset('tlist', shape=(N*1024,), 
                                 dtype=[('TriggerNo', np.uint32), ('ChannelID', np.uint32), 
                                        ('t_s', np.float16), ('q_s', np.float32)], chunks=True)
    s_history = opt.create_dataset('s_history', shape=(N*TRIALS*100,), 
                                   dtype=d_history, chunks=True)
                                   
    metropolis(ent, sample, mu0, d_tlist, s_history)
    print('The output file path is {}'.format(fopt))

print('Prediction generated, real time {0:.02f}s, cpu time {1:.02f}s'.format(time.time() - tic, time.process_time() - cpu_tic))

print('Finished! Consuming {0:.02f}s in total, cpu time {1:.02f}s.'.format(time.time() - global_start, time.process_time() - cpu_global_start))
