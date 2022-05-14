#!/usr/bin/env python3
import time
import argparse

import h5py
import numpy as np
import cupy as cp

from scipy.stats import norm, exponnorm, erlang

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest="opt", type=str, help="output file")
psr.add_argument("ipt", type=str, help="input file")
psr.add_argument("--ref", type=str, help="truth file")
psr.add_argument("--size", type=int, default=100, help="batch size")
args = psr.parse_args()

fipt = args.ipt
fopt = args.opt

TRIALS = 5000

#profile
def vcombine(A, cx, t, w_all):
    '''
    t is 2 x l_e
    '''
    frac, ti = cp.modf(t)
    ti = cp.array(ti + 1, np.int32) % A.shape[2] # avoid ti + 1 to overflow
    A_vec = (1 - frac)[:, :, None] * A[w_all, :, ti-1] + frac[:, :, None] * A[w_all, :, ti]
    c_vec = (1 - frac)[:, :, None] * cx[w_all, :, ti-1] + frac[:, :, None] * cx[w_all, :, ti]
    return A_vec, c_vec

vstep = cp.array((-1, 1), np.float32)

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
    Δcx = -cp.transpose(beta @ c_vec, (0, 2, 1)) @ (c_vec @ A)
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

#profile
def v_rt(_s, _ts, w_all):
    frac, ti = np.modf(_s)
    ti = np.array(ti, np.int32)
    return (1 - frac) * _ts[w_all, ti] + frac * _ts[w_all, ti+1]

with h5py.File(args.ref, 'r', libver='latest', swmr=True) as ipt:
    tau = ipt['Readout/Waveform'].attrs['tau'].item()
    mu_true = ipt["Readout/Waveform"].attrs["mu"]
    sigma = ipt['Readout/Waveform'].attrs['sigma'].item()
    start = ipt['SimTruth/T'][:]

b_t0 = [0., 600.]

if tau == 0.0:
    sel_lc = cp.ElementwiseKernel(
        'float32 t, bool sel',
        'float32 lprob',
        f"lprob = sel ? -log({sigma}) - 0.5 * log(2.0 * {np.pi}) - 0.5 * (t / {sigma}) * (t / {sigma}) : 0",
        'sel_lc')
    lcs = norm(scale = sigma)
elif sigma == 0.0:
    sel_lc = cp.ElementwiseKernel(
        'float32 t, bool sel',
        'float32 lprob',
        f"lprob = sel ? (t > 0 ? -log({tau}) - t / {tau} : -1e4) : 0",
        'sel_lc')
else:
    alpha = 1 / tau
    co = -np.log(2.0 * tau) + alpha * alpha * sigma * sigma / 2.0
    ass = alpha * sigma * sigma
    s2s = np.sqrt(2.0) * sigma
    sel_lc = cp.ElementwiseKernel(
        'float32 t, bool sel',
        'float32 lprob',
        f"lprob = sel ? {co} + log(1.0 - erf(({ass} - t) / {s2s})) - {alpha} * t : 0",
        'sel_lc')
    lcs = exponnorm(K = tau/sigma, scale = sigma)

sel_add = cp.ElementwiseKernel('float32 delta, bool sel', 'float32 dest', "if(sel) dest += delta", "sel_add")
sel_add_2D = cp.RawKernel(r'''
        extern "C" __global__
        void sel_add_2D(const float *delta, const bool* sel, float *dest, int batchsize, int N) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            for(int i=0;i<batchsize;i++)
            {
                if(sel[i]) dest[i*N+tid] += delta[i*N+tid];
            }
        }
''', 'sel_add_2D')
sel_assign = cp.ElementwiseKernel('float32 value, bool sel', 'float32 dest', "if(sel) dest = value", "sel_assign")

#profile
def batch(A, cx, index, tq, t_index, s, z, t0_min=100, t0_max=500):
    """
    batch
    ====
    连续时间游走
    cx: Cov^-1 * A, 详见 FBMP
    s: list of PE locations
    z: residue waveform (raw - predicted)
    t_index: 从实际时间到 tq["t_s"] 坐标时间的映射

    home_s 是在时间意义上的连续变量，从 light curve 抽样

    0   1   2   3   4   5   6   7   8   9   l_t = 10
    +---+---+---+---+---+---+---+---+---+
    0  0.1 0.2 0.2 0.5 0.4 0.3 0.2  0   0   q_s
    0  0.1 0.3 0.5 1.0 1.4 1.7 1.9 1.9 1.9  cq
     _h \in [0, 9) 范围的测度是 9

    int(_h) 是插值的下指标 占比为 1 - decimal(_h)
    int(_h) + 1 是插值的上指标 占比为 decimal(_h)
    """
    sig2w = cp.asarray(index["sig2w"], np.float32)
    sig2s = cp.asarray(index["sig2s"], np.float32)
    mus = cp.asarray(index["mus"], np.float32)
    NPE = index["NPE"]

    a0 = A[:, :, 0]
    b0 = 1 + sig2s * (a0[:, None, :] @ a0[:, :, None]).squeeze() / sig2w

    l_e = len(index)
    l_s = s.shape[1]
    w_all = cp.arange(l_e) # index of all the waveforms
    nw_all = np.arange(l_e)

    l_t = tq.shape[1]
    # istar [0, 1) 之间的随机数，用于点中 PE
    istar = np.random.rand(l_e, TRIALS) # 同时可用于创生位置的选取

    home_rt = lcs.rvs(size = istar.shape)
    ts_min = tq['t_s'][:, 0]
    ts_length = tq['t_s'][np.arange(l_e), index["l_t"]-1] - ts_min

    t0 = np.zeros(l_e, np.float32)
    e_hit = NPE > 0

    t0[e_hit] = v_rt(s[e_hit, 0], tq["t_s"], e_hit)
    e_nonhit = ~e_hit
    t0[e_nonhit] = tq["t_s"][e_nonhit, 0]
    t0 = np.clip(t0, t0_min, t0_max)

    # step: +1 创生一个 PE， -1 消灭一个 PE， +2 向左或向右移动
    flip = np.random.choice((-1, 1, 2), (l_e, TRIALS), p=np.array((1, 1, 2)) / 4)
    Δν_g = cp.zeros(l_e, dtype=np.float32)
    Δν = np.zeros(l_e, dtype=np.float32)
    ν = np.zeros(l_e, dtype=np.float32)
    ν_max = np.zeros(l_e, dtype=np.float32)
    Δν_history = np.zeros((l_e, TRIALS), dtype=np.float32) # list of Δν's
    t0_history = np.zeros((l_e, TRIALS), dtype=np.float32)
    s0_history = np.zeros((l_e, TRIALS), dtype=np.uint32) # 0-norm of s
    mu_history = np.zeros((l_e, TRIALS), dtype=np.float32) # 0-norm of s
    s_history = np.zeros((l_e, TRIALS * l_s), dtype=np.float32)

    # t0_accept = 0
    loc = np.zeros((l_e, 2)) # float64
    fsig2s_inv = cp.diag(vstep)[None, :, :] * (1/sig2s)[:, None, None]
    det_fsig2s_inv = cp.linalg.det(fsig2s_inv)
    fmu = vstep[None, :] * mus[:, None]

    last_max_s = s.copy()
    s_max_index = np.zeros(l_e, dtype=np.uint32)

    for i, (t, step, home, wander, wt, accept, acct) in enumerate(
        zip(
            istar.T,
            flip.T,
            home_rt.T,
            np.random.normal(size=(TRIALS, l_e)),
            np.random.normal(scale=10, size=(TRIALS, l_e)),
            *np.log(np.random.rand(2, TRIALS, l_e)),
        )
    ):
        if i % 1000 == 0:
            print(i)

        mNPE = np.max(NPE)
        rt = v_rt(s[:, :mNPE], tq["t_s"], np.arange(l_e)[:, None])
        nt0 = t0 + wt.astype(np.float32)
        sel = cp.asarray(np.arange(mNPE)[None, :] < NPE[:, None])
        lc0 = cp.sum(sel_lc(cp.asarray(rt - t0[:, None]), sel), axis=1)
        lc1 = cp.sum(sel_lc(cp.asarray(rt - nt0[:, None]), sel), axis=1)
        t0_acceptance = np.logical_and(cp.asnumpy(lc1 - lc0) >= acct, np.logical_and(nt0 >= t0_min, nt0 <= t0_max))
        # t0_accept += np.sum(t0_acceptance) / len(t0_acceptance)
        np.putmask(t0, t0_acceptance, nt0)
        t0_history[:, i] = t0

        ### mu 直接采样分布：e^{-\mu/\mu_p} * e^{-\mu} * \mu^N
        # Erlang 分布 => scale = (\mu_0 + 1) / \mu_0, a = N + 1
        log_mu = np.log(erlang.rvs(a = NPE+1, scale = mu_true / (mu_true + 1)))

        ### 光变曲线和移动计算
        Δν[:] = 0
        e_bounce = NPE == 0
        step[e_bounce] = 1
        accept[e_bounce] += np.log(4)  # 惩罚
        ea_bounce = np.logical_and(NPE == 1, step == -1)
        accept[ea_bounce] -= np.log(4) # 1 -> 0: 行动后从 0 脱出的几率大，需要鼓励

        e_create = step == 1
        e_minus = ~e_create
        create_rt = t0[e_create] + home[e_create] - ts_min[e_create]
        acc_e_create = np.logical_and(create_rt >= 0, create_rt <= ts_length[e_create])
        e_create[nw_all[e_create][~acc_e_create]] = False
        create_i = v_rt(create_rt[acc_e_create], t_index, e_create) # from time to index.

        e_move = step == 2
        e_annihilate = step == -1
        e_pm = np.logical_and(e_create, e_annihilate)
        loc[e_create, 0] = l_t # 0 A_vec
        op = np.array(t * NPE, dtype=np.int32)
        loc[e_minus, 0] = s[e_minus, op[e_minus]] # annihilate + move
        loc[e_move, 1] = periodic(loc[e_move, 0] + wander[e_move], index["l_t"][e_move])
        loc[e_create, 1] = create_i
        loc[e_annihilate, 1] = l_t
        
        ### 矩阵 Δν 计算, 即使无意义的 e_create 也一起计算，否则有较大 GPU 性能损失
        # 原则：fancy index 只在 CPU 上使用。
        vA, vc = vcombine(A, cx, cp.asarray(loc, np.float32), w_all[:, None])
        Δν_g, beta = vmove1(vA, vc, z, fmu, fsig2s_inv, det_fsig2s_inv, b0)

        ## non-move cases, step == 1, -1
        NPE[e_create] += 1
        Δν[e_pm] += step[e_pm] * (log_mu[e_pm] - np.log(NPE[e_pm]))
        NPE[e_create] -= 1
        ########
        Δν += cp.asnumpy(Δν_g)
        #######

        ### 计算 Δcx, Δz, 更新 cx 和 z。对 accept 进行特别处理
        e_accept = np.logical_and(Δν >= accept, np.logical_or(e_create, e_minus))
        _e_accept = cp.asarray(e_accept)
        Δcx, Δz = vmove2(vA, vc, fmu, A, beta)
        Shape = cx.shape
        sel_add_2D((Shape[1],), (Shape[2],), (Δcx, _e_accept, cx, Shape[0], Shape[1] * Shape[2]))
        sel_add(Δz, _e_accept[:, None], z)
        ########

        # 增加
        ea_create = np.logical_and(e_accept, e_create)

        s[ea_create, NPE[ea_create]] = loc[ea_create, 1]
        NPE[ea_create] += 1
        # 减少
        ea_annihilate = np.logical_and(e_accept, step == -1)

        NPE[ea_annihilate] -= 1
        s[ea_annihilate, op[ea_annihilate]] = s[ea_annihilate, NPE[ea_annihilate]]
        # 移动
        ea_move = np.logical_and(e_accept, e_move)
        s[ea_move, op[ea_move]] = loc[ea_move, 1]

        Δν[~e_accept] = 0
        step[~e_accept] = 0

        ν += Δν
        breakthrough = ν > ν_max
        ν_max[breakthrough] = ν[breakthrough]
        s_max_index[breakthrough] = i
        last_max_s[breakthrough] = s[breakthrough]
        Δν_history[:, i] = Δν
        flip[:, i] = step
        s0_history[:, i] = NPE
        mu_history[:, i] = log_mu
        s_history[:, i*l_s:(i+1)*l_s] = s
    # print("t0 acceptance:", t0_accept / TRIALS)
    # Assign one PE to the waveform without any PE
    s_init = np.full(l_s, 0.)
    s_init[0] = 1.
    last_max_s[last_max_s.sum(axis=1) == 0] = s_init
    return flip, s0_history, t0_history, Δν_history, s_max_index, last_max_s, s_history, mu_history


with h5py.File(fipt, "r", libver="latest", swmr=True) as ipt:
    A = ipt["A"][:]
    cx = ipt["cx"][:]
    index = ipt["index"][:]
    s = ipt["s"][:]
    tq = ipt["tq"][:]
    z = ipt["z"][:]
    t_index = ipt["t_index"][:]

l_e = len(index)
s_t = np.argsort(index["l_t"])

sample = np.zeros((l_e * TRIALS), dtype=[("TriggerNo", "u4"), ("ChannelID", "u4"),
                                         ("flip", "i2"), ("s0", "u4"), ("t0", "f8"),
                                         ("mu", "f8"), ("delta_nu", "f4")])
sample["TriggerNo"] = np.repeat(index["TriggerNo"], TRIALS)
sample["ChannelID"] = np.repeat(index["ChannelID"], TRIALS)
s_max = np.zeros(l_e, dtype=[("TriggerNo", "u4"),
                             ("ChannelID", "u4"),
                             ("s_max_index", "u4"),
                             ("s_max", "f4", index['l_t'].max()),
                             ("consumption", "f8"),])
s_max["TriggerNo"] = index["TriggerNo"]
s_max["ChannelID"] = index["ChannelID"]

for part in range(l_e // args.size + 1):
    time_start = time.time()
    i_part = s_t[part * args.size:(part + 1) * args.size]
    l_part = len(i_part)
    if l_part:
        ind_part = index[i_part]
        lp_t = np.max(ind_part["l_t"])
        lp_interval = np.max(ind_part["l_interval"])
        lp_wave = np.max(ind_part["l_wave"])
        lp_NPE = np.max(ind_part["NPE"])
        print(lp_t, lp_wave, lp_NPE)

        null = np.zeros((l_part, lp_wave, 1), np.float32) # cx, A[:, :, -1] = 0  用于 +- 的空白维度
        s_null = np.zeros((l_part, lp_NPE * 2), np.float32) # 富余的 PE 活动空间
        (flip, s0_history, t0_history, Δν_history, s_max_index, last_max_s, loc, mu_history
         ) = batch(cp.asarray(np.append(A[i_part, :lp_wave, :lp_t], null, axis=2), np.float32),
                   cp.asarray(np.append(cx[i_part, :lp_wave, :lp_t], null, axis=2), np.float32),
                   index[i_part], 
                   tq[i_part, :lp_t],
                   t_index[i_part, :lp_interval],
                   np.append(s[i_part, :lp_NPE], s_null, axis=1),
                   cp.asarray(z[i_part, :lp_wave], np.float32))
        fi_part = (i_part * TRIALS)[:, None] + np.arange(TRIALS)[None, :]
        fip = fi_part.flatten()
        sample["flip"][fip] = flip.flatten()
        sample["s0"][fip] = s0_history.flatten()
        sample["t0"][fip] = t0_history.flatten()
        sample["mu"][fip] = mu_history.flatten()
        sample["delta_nu"][fip] = Δν_history.flatten()
        s_max["s_max_index"][i_part] = s_max_index
        min_col = min(last_max_s.shape[1], index['l_t'].max())
        s_max["s_max"][i_part, :min_col] = last_max_s[:, :min_col]
        s_max["consumption"][i_part] = (time.time() - time_start) / l_part

with h5py.File(fopt, "w") as opt:
    opt.create_dataset("sample", data=sample, compression="gzip", shuffle=True)
    opt.create_dataset("s_max", data=s_max, compression="gzip", shuffle=True)
