#!/usr/bin/env python3
'''
Replay FSMP to inspect the Markov process.

'''
import argparse
import pandas as pd
import h5py
import numpy as np
from math import erf
np.seterr("raise")

psr = argparse.ArgumentParser()
psr.add_argument("ipt", type=str, help="input file for the markov chain")
psr.add_argument("--sparse", type=str, help="sparsified file from LucyDDM preparation")
psr.add_argument("--trigger", type=int, help="Select trigger number")
args = psr.parse_args()

sample = pd.read_hdf(args.ipt, "sample").query(f"TriggerNo=={args.trigger}")
jumps = sample.query("flip != 0")
with h5py.File(args.sparse, "r") as sparse:
    index0 = sparse["index"][:]
    loc = np.searchsorted(index0["TriggerNo"], args.trigger)

    index = sparse["index"][loc]
    l_t = index["l_t"]
    l_wave = index["l_wave"]

    A = sparse["A"][loc, :l_wave, :l_t]
    cx = sparse["cx"][loc, :l_wave, :l_t]
    tq = sparse["tq"][loc, :l_t]
    z = sparse["z"][loc, :l_wave]
    s0 = sparse["s"][loc]

NPE = round(index["mu0"])
cq = tq["cq"][:l_t]

### copied from metropolis.py
def combine(A, cx, t):
    frac, ti = np.modf(t)
    ti = int(ti)
    alpha = np.array((1 - frac, frac))
    return alpha @ A[:, ti : (ti + 2)].T, alpha @ cx[:, ti : (ti + 2)].T

def move1(A_vec, c_vec, z, step, mus, sig2s):
    fsig2s = step * sig2s
    beta_under = 1 + fsig2s * np.dot(A_vec, c_vec)
    beta = fsig2s / beta_under
    Δν = 0.5 * (beta * (z @ c_vec + mus / sig2s) ** 2 - mus ** 2 / fsig2s)
    print(1/beta_under)
    Δν -= 0.5 * np.log(beta_under)  # space
    return Δν, beta

def move2(A_vec, c_vec, step, mus, A, beta):
    Δcx = -np.einsum("n,m,mp->np", beta * c_vec, c_vec, A, optimize=True)
    Δz = -step * A_vec * mus
    return Δcx, Δz

def move(A_vec, c_vec, z, step, mus, sig2s, A):
    Δν, beta = move1(A_vec, c_vec, z, step, mus, sig2s)
    return Δν, *move2(A_vec, c_vec, step, mus, A, beta)

a0 = A[:, 0]
b0 = 1 + index["sig2s"] * a0 @ a0 / index["sig2w"]

vstep = np.array([-1, 1], np.float32)
def vmove1(A_vec, c_vec, z, mus, sig2s):
    '''
    A_vec: 2 x l_wave
    c_vec: 2 x l_wave
    '''
    fsig2s_inv = np.diag(vstep) / sig2s
    ac = A_vec @ c_vec.T
    beta_inv = fsig2s_inv + (ac + ac.T)/2
    beta = np.linalg.inv(beta_inv)

    fmu = mus * vstep
    zc = np.einsum('jk,k->j', c_vec, z) + np.einsum('j,jk->k', fmu, fsig2s_inv)
    Δν = 0.5 * np.einsum('j,jk,k->', zc, beta, zc)
    logee = -np.linalg.det(beta) / sig2s ** 2
    print(logee)
    Δν += 0.5 * np.log(logee) # det(diag(-sig2s, sig2s)) = sig2s**2
    return Δν, beta, fmu

def vmove2(A_vec, c_vec, fmu, A, beta):
    Δcx = -np.einsum("ij,iv,jw,wt->vt", beta, c_vec, c_vec, A, optimize=True)
    Δz = -np.einsum("j,jw->w", fmu, A_vec)
    return Δcx, Δz

def lc(x, tau=20, sigma=5):
    """
    light curve
    """
    alpha = 1 / tau
    co = -np.log(2.0 * tau) + alpha * alpha * sigma * sigma / 2.0
    x_erf = (alpha * sigma * sigma - x) / (np.sqrt(2.0) * sigma)
    return co + np.log(1.0 - erf(x_erf)) - alpha * x

def v_rt(_s, _ts):
    frac, ti = np.modf(_s)
    ti = int(ti)
    return (1 - frac) * _ts[ti] + frac * _ts[ti+1]

cx0 = np.copy(cx)
z0 = np.copy(z)
frac, ti = np.modf(s0[:NPE])
ti = np.array(ti, np.int_)
vA0 = (1-frac) * A[:, ti] + frac * A[:, ti+1]
z_null = z0 + np.sum(vA0, axis=1) * index["mus"]

def nu(s):
    frac, ti = np.modf(s)
    ti = np.array(ti, np.int_)
    A_vec = (1-frac) * A[:, ti] + frac * A[:, ti+1]
    phi = (A_vec @ A_vec.T) * index["sig2s"] + np.eye(A.shape[0]) * index["sig2w"]
    phi_inv = np.linalg.inv(phi)
    z = z_null - np.sum(A_vec * index["mus"], axis=1)
    return -0.5 * z @ phi_inv @ z - 0.5 * np.log(np.linalg.det(phi)), phi_inv, z

s = np.zeros(NPE*8)
s[:NPE] = s0[:NPE]
precise_nu0, phi_inv, z = nu(s[:NPE])
precise_cx = phi_inv @ A
print(np.linalg.norm(cx - cx0))

for idx, jump in jumps.iterrows():
    t_minus = jump["annihilation"]
    t_plus = jump["creation"]
    sign = f"m: {t_minus:.2f} -> {t_plus:.2f}"
    t0 = jump["t0"]
    s_raw = np.copy(s)
    if np.abs(jump["flip"]-2) < 1e-6:
        A_d, c_d = combine(A, cx, t_minus)
        A_u, c_u = combine(A, cx, t_plus)
        vA_move = np.stack((A_d, A_u), axis=0)
        vc_move = np.stack((c_d, c_u), axis=0)
        Δν, beta, fmu = vmove1(vA_move, vc_move, z, index["mus"], index["sig2s"])
        Δν_l = -lc(v_rt(t_minus, tq["t_s"]) - t0)
        Δν_l += lc(v_rt(t_plus, tq["t_s"]) - t0)
        Δcx, Δz = vmove2(vA_move, vc_move, fmu, A, beta)
        cx += Δcx
        z += Δz
        op = np.where(np.abs(s[:NPE] - t_minus) < 1e-6)[0]
        assert len(op) == 1
        s[op] = t_plus
    else: # +-1
        if jump["flip"] < 0:
            t, sign = (t_minus, f'-: {t_minus:.2f}')
            op = np.where(np.abs(s[:NPE] - t_minus) < 1e-6)[0]
            assert len(op) == 1
            n_NPE = NPE
            NPE-=1
            s[op] = s[NPE]
        else:
            t, sign = (t_plus, f'+: {t_plus:.2f}')
            s[NPE] = t_plus
            NPE+=1
            n_NPE = NPE
        A_vec, c_vec = combine(A, cx, t)
        Δν, beta = move1(A_vec, c_vec, z, jump["flip"], index["mus"], index["sig2s"])
        Δν_l = jump["flip"] * lc(v_rt(t, tq["t_s"]) - t0)
        log_mu = np.log(index["mu0"])
        Δν_l += jump["flip"] * (log_mu - np.log(tq["q_s"][int(t) + 1]) - np.log(n_NPE))
        Δcx, Δz = move2(A_vec, c_vec, jump["flip"], index["mus"], A, beta)
        cx += Δcx
        z += Δz
    precise_nu = nu(s[:NPE])[0]
    ref_nu = jump["delta_nu"]
    print(f"{precise_nu:.1f} {precise_nu - precise_nu0 + Δν_l:.4f} {Δν + Δν_l:.4f} ref: {ref_nu:.2f} | {sign}")
    precise_nu0 = precise_nu
