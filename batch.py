#!/usr/bin/env python3
import time
import argparse

import h5py
import numpy as np
import cupy as cp
from numba import njit, vectorize

from math import erf
from scipy.stats import norm

import itertools as it

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("ipt", type=str, help="input file")
psr.add_argument("--size", type=int, default=100, help="batch size")
args = psr.parse_args()

fipt = args.ipt
fopt = args.opt

TRIALS = 5000

def vcombine(A, cx, t, w_all):
    '''
    t is 2 x l_e
    '''
    frac, ti = cp.modf(t)
    ti = cp.array(ti, np.int32)
    A_vec = (1 - frac)[:, :, None] * A[w_all, :, ti] + frac[:, :, None] * A[w_all, :, ti+1]
    c_vec = (1 - frac)[:, :, None] * cx[w_all, :, ti] + frac[:, :, None] * cx[w_all, :, ti+1]
    return A_vec, c_vec

@vectorize(cache=True)
def verf(x):
    return erf(x)

@njit(nogil=True, cache=True)
def lc(x, tau=20, sigma=5):
    """
    light curve
    """
    alpha = 1 / tau
    co = -np.log(2.0 * tau) + alpha * alpha * sigma * sigma / 2.0
    x_erf = (alpha * sigma * sigma - x) / (np.sqrt(2.0) * sigma)
    return co + np.log(1.0 - verf(x_erf)) - alpha * x

vstep = cp.array((-1, 1), np.float64)

#profile
def vmove1(A_vec, c_vec, z, fmu, fsig2s_inv, det_fsig2s_inv, b0):
    '''
    A_vec: 行向量 x 2
    c_vec: 行向量 x 2
    vstep: 构造 (-1, 1) 则 np.einsum('ej,ejk,ek->e', fmu, fsig2s_inv, fmu) 恒等于 0

    cp.einsum('eiw,ejw->eij', A_vec, c_vec)

    cp.einsum('ejk,ek->ej', c_vec, z) + cp.einsum('ej,ejk->ek', fmu, fsig2s_inv)
    cp.einsum('ej,ejk,ek->e', zc, beta, zc)
    '''
    ac = A_vec @ cp.transpose(c_vec, (0, 2, 1))
    beta_inv = fsig2s_inv + (ac + cp.transpose(ac, (0, 2, 1))) / 2
    beta = cp.linalg.inv(beta_inv)

    zc = (c_vec @ z[:, :, None]).squeeze() + (fmu[:, None, :] @ fsig2s_inv).squeeze()
    Δν = 0.5 * (zc[:, None, :] @ beta @ zc[:, :, None]).squeeze()

    Δν += 0.5 * cp.log(cp.clip(cp.linalg.det(beta) * det_fsig2s_inv, 1/b0, b0))
    return Δν, beta

#profile
def vmove2(A_vec, c_vec, fmu, A, beta):
    '''
    "eij, eiv, ejw, ewt->evt"
    beta c_vec c_vec A
    '''
    Δcx = -np.transpose(beta @ c_vec, (0, 2, 1)) @ (c_vec @ A)
    Δz = -(fmu[:, None, :] @ A_vec).squeeze()
    return Δcx, Δz

def periodic(_h, _lt):
    '''
    enforce a periodic boundary at 0 and _lt - 1
    '''
    c_overflow = _h < 0
    _h[c_overflow] += _lt[c_overflow] - 1
    c_overflow = _h > _lt - 1
    _h[c_overflow] -= _lt[c_overflow] - 1
    return _h

def v_rt(_s, _ts):
    frac, ti = np.modf(_s)
    ti = np.array(ti, np.int32)
    w_all = np.arange(_ts.shape[0]) # index of all the waveforms
    return (1 - frac) * _ts[w_all, ti] + frac * _ts[w_all, ti+1]

@njit(nogil=True, cache=True)
def shift(t0, nt0, si, s, ts, NPE, acct):
    '''
    s: PE time
    si: integer part of PE time
    return t0 for the next step
    '''
    for e, (_s, _ti, _ts, _npe, _t0, _nt0, _acct) in enumerate(zip(s, si, ts, NPE, t0, nt0, acct)):
        _s = _s[:_npe]
        _ti = _ti[:_npe]
        frac = _s - _ti
        _rt = (1 - frac) * _ts[_ti] + frac * _ts[_ti+1]
        if np.sum(lc(_rt - _nt0)) - np.sum(lc(_rt - _t0)) >= _acct:
            t0[e] = _nt0

sel_add =  cp.ElementwiseKernel('float64 delta, bool sel', 'float64 dest', "if(sel) dest += delta", "mask_add")
#profile
def batch(A, cx, index, tq, s, z):
    """
    batch
    ====
    连续时间游走
    cx: Cov^-1 * A, 详见 FBMP
    s: list of PE locations
    mu_t: LucyDDM 的估算 PE 数
    z: residue waveform (raw - predicted)

    home_s 是在指标意义上的连续变量

    0   1   2   3   4   5   6   7   8   9   l_t = 10
    +---+---+---+---+---+---+---+---+---+
    0  0.1 0.2 0.2 0.5 0.4 0.3 0.2  0   0   q_s
    0  0.1 0.3 0.5 1.0 1.4 1.7 1.9 1.9 1.9  cq
     _h \in [0, 9) 范围的测度是 9

    int(_h) 是插值的下指标 占比为 1 - decimal(_h)
    int(_h) + 1 是插值的上指标 占比为 decimal(_h)
    """
    sig2w = cp.asarray(index["sig2w"])
    sig2s = cp.asarray(index["sig2s"])
    mus = cp.asarray(index["mus"])

    a0 = A[:, :, 0]
    b0 = 1 + sig2s * (a0[:, None, :] @ a0[:, :, None]).squeeze() / sig2w

    l_e = len(index)
    w_all = cp.arange(l_e) # index of all the waveforms
    w_all2 = cp.stack((w_all, w_all), axis=1) # l_e x 2 index of all the waveforms

    l_t = tq.shape[1]
    # istar [0, 1) 之间的随机数，用于点中 PE
    istar = np.random.rand(l_e, TRIALS) # 同时可用于创生位置的选取

    # 根据 p_cha 采样得到的 PE 序列。供以后的产生过程使用。这两行是使用了 InverseCDF 算法进行的MC采样。
    fp = np.arange(l_t)
    home_s = np.array([ np.interp(_is, xp=_xp[:_lt], fp=fp[:_lt]) for _is, _xp, _lt in zip(istar, tq["cq"], index["l_t"])])

    mNPE = np.max(NPE)

    t0 = np.zeros(l_e, np.float32)
    e_hit = NPE > 0
    t0[e_hit] = v_rt(s[e_hit, 0], tq["t_s"][e_hit])
    e_nonhit = ~e_hit
    t0[e_nonhit] = tq["t_s"][e_nonhit, 0]

    # step: +1 创生一个 PE， -1 消灭一个 PE， +2 向左或向右移动
    flip = np.random.choice((-1, 1, 2), (l_e, TRIALS), p=np.array((1, 1, 2)) / 4)
    Δν_g = cp.zeros(l_e, dtype=np.float32)
    Δν = np.zeros(l_e, dtype=np.float32)
    Δν_history = np.zeros((l_e, TRIALS), dtype=np.float32) # list of Δν's
    annihilations = np.zeros((l_e, TRIALS)) # float64
    creations = np.zeros((l_e, TRIALS)) # float64
    t0_history = np.zeros((l_e, TRIALS), dtype=np.float32)
    s0_history = np.zeros((l_e, TRIALS), dtype=np.uint32) # 0-norm of s

    log_mu = np.log(index["mu0"])  # 猜测的 Poisson 流强度
    loc = np.zeros((l_e, 2)) # float64
    fsig2s_inv = cp.diag(vstep)[None, :, :] * (1/sig2s)[:, None, None]
    det_fsig2s_inv = cp.linalg.det(fsig2s_inv)
    fmu = vstep[None, :] * mus[:, None]

    for i, (t, step, home, wander, wt, accept, acct) in enumerate(
        zip(
            istar.T,
            flip.T,
            home_s.T,
            *np.random.normal(size=(2, TRIALS, l_e)),
            *np.log(np.random.rand(2, TRIALS, l_e)),
        )
    ):
        if i % 1000 == 0:
            print(i)

        s_t0 = s[:, :np.max(NPE)]
        shift(t0, t0 + wt, np.array(s_t0, np.int32), s_t0, tq["t_s"], NPE, acct)
        t0_history[:, i] = t0

        ### 光变曲线和移动计算
        Δν[:] = 0
        e_bounce = NPE == 0
        step[e_bounce] = 1
        accept[e_bounce] += np.log(4)  # 惩罚
        ea_bounce = np.logical_and(NPE == 1, step == -1)
        accept[ea_bounce] -= np.log(4) # 1 -> 0: 行动后从 0 脱出的几率大，需要鼓励

        e_create = step == 1
        e_minus = ~e_create
        e_move = step == 2
        e_pm = ~e_move
        e_annihilate = step == -1
        e_plus = ~e_annihilate
        loc[e_create, 0] = l_t # 0 A_vec
        op = np.array(t * NPE, dtype=np.int32)
        loc[e_minus, 0] = s[e_minus, op[e_minus]] # annihilate + move
        loc[e_move, 1] = periodic(loc[e_move, 0] + wander[e_move], index["l_t"][e_move])
        loc[e_create, 1] = periodic(home[e_create], index["l_t"][e_create])

        ## -1 cases, step == 2, -1
        Δν[e_minus] -= lc(v_rt(loc[e_minus, 0], tq["t_s"][e_minus]) - t0[e_minus])

        ## +1 cases, step == 2, 1
        Δν[e_plus] += lc(v_rt(loc[e_plus, 1], tq["t_s"][e_plus]) - t0[e_plus])

        ## non-move cases, step == 1, -1
        NPE[e_create] += 1
        loc[e_annihilate, 1] = loc[e_annihilate, 0]
        Δν[e_pm] += step[e_pm] * (log_mu[e_pm] - np.log(tq["q_s"][e_pm, np.array(loc[e_pm, 1], dtype=np.int32) + 1]) - np.log(NPE[e_pm]))
        loc[e_annihilate, 1] = l_t
        NPE[e_create] -= 1
        ########

        ### 矩阵 Δν 计算
        vA, vc = vcombine(A, cx, cp.asarray(loc), w_all2)
        Δν_g, beta = vmove1(vA, vc, z, fmu, fsig2s_inv, det_fsig2s_inv, b0)
        Δν += cp.asnumpy(Δν_g)
        #######

        ### 计算 Δcx, Δz, 更新 cx 和 z。对 accept 进行特别处理
        e_accept = Δν >= accept
        _e_accept = cp.asarray(e_accept)
        Δcx, Δz = vmove2(vA, vc, fmu, A, beta)
        sel_add(Δcx, _e_accept[:, None, None], cx)
        sel_add(Δz, _e_accept[:, None], z)
        ########

        # 增加
        ea_create = np.logical_and(e_accept, e_create)
        ea_plus = np.logical_and(e_accept, e_plus)
        creations[ea_plus, i] = loc[ea_plus, 1]

        s[ea_create, NPE[ea_create]] = loc[ea_create, 1]
        NPE[ea_create] += 1
        # 减少
        ea_annihilate = np.logical_and(e_accept, step == -1)
        ea_minus = np.logical_and(e_accept, e_minus)
        annihilations[ea_minus, i] = loc[ea_minus, 0]

        NPE[ea_annihilate] -= 1
        s[ea_annihilate, op[ea_annihilate]] = s[ea_annihilate, NPE[ea_annihilate]]
        # 移动
        ea_move = np.logical_and(e_accept, e_move)
        s[ea_move, op[ea_move]] = loc[ea_move, 1]

        Δν[~e_accept] = 0
        step[~e_accept] = 0

        Δν_history[:, i] = Δν
        flip[:, i] = step
        s0_history[:, i] = NPE
    return flip, s0_history, t0_history, Δν_history, annihilations, creations

with h5py.File(fipt, "r", libver="latest", swmr=True) as ipt:
    A = ipt["A"][:]
    cx = ipt["cx"][:]
    index = ipt["index"][:]
    s = ipt["s"][:]
    tq = ipt["tq"][:]
    z = ipt["z"][:]

l_e = len(index)
s_t = np.argsort(index["l_t"])

sample = np.zeros((l_e * TRIALS), dtype=[("TriggerNo", "u4"), ("ChannelID", "u4"),
                                         ("flip", "i2"), ("s0", "u4"), ("t0", "f8"), 
                                         ("annihilation", "f8"), ("creation", "f8"), 
                                         ("delta_nu", "f4")])
sample["TriggerNo"] = np.repeat(index["TriggerNo"], TRIALS)
sample["ChannelID"] = np.repeat(index["ChannelID"], TRIALS)

for part in range(l_e // args.size + 1):
    i_part = s_t[part * args.size:(part + 1) * args.size]
    l_part = len(i_part)
    if l_part:
        ind_part = index[i_part]
        lp_t = np.max(ind_part["l_t"])
        lp_wave = np.max(ind_part["l_wave"])
        lp_NPE = np.max(ind_part["NPE"])
        print(lp_t, lp_wave, lp_NPE)

        null = np.zeros((l_part, lp_wave, 1)) # cx, A[:, :, -1] = 0  用于 +- 的空白维度
        s_null = np.zeros((l_part, lp_NPE)) # 富余的 PE 活动空间
        (flip, s0_history, t0_history, Δν_history, annihilations, creations,
         ) = batch(cp.asarray(np.append(A[i_part, :lp_wave, :lp_t], null, axis=2), np.float64),
                   cp.asarray(np.append(cx[i_part, :lp_wave, :lp_t], null, axis=2), np.float64),
                   index[i_part], tq[i_part, :lp_t], 
                   np.append(s[i_part, :lp_NPE], s_null, axis=1),
                   cp.asarray(z[i_part, :lp_wave], np.float64))
        fi_part = (i_part * TRIALS)[:, None] + np.arange(TRIALS)[None, :]
        fip = fi_part.flatten()
        sample["flip"][fip] = flip.flatten()
        sample["s0"][fip] = s0_history.flatten()
        sample["t0"][fip] = t0_history.flatten()
        sample["annihilation"][fip] = annihilations.flatten()
        sample["creation"][fip] = creations.flatten()
        sample["delta_nu"][fip] = Δν_history.flatten()

with h5py.File(fopt, "w") as opt:
    opt.create_dataset("sample", data=sample, compression="gzip", shuffle=True)
