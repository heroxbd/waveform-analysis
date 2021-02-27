import math

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import scipy.special as special

M = 128
N = 512
p1 = 0.04
vnc = 0
SNRdB = 15
D = 20
stop = 0

sig2s = np.array([0., 40.**2])
mus = np.array([0., 160.])

Q = len(mus) - 1
ps = np.insert(np.ones(Q) * p1 / Q, 0, 1 - p1)
intps = np.cumsum(np.insert(ps, 0, 0))
sig2s_tmp = np.hstack([sig2s, sig2s[1] * np.ones(Q - 1)])
s_true = np.zeros(N).astype(int)
if vnc != 0:
    v = np.random.rand(N)
    for i in range(2, len(ps) + 1):
        s_true = s_true + (v > intps[i])
else:
    while sum(s_true > 0) < round(p1 * N):
        s_true[math.floor(np.random.rand() * N)] = math.ceil(np.random.rand() * Q)

A = np.random.randn(M, N)
A = np.matmul(A, np.diag(1. / np.sqrt(np.diag(np.matmul(A.T, A)))))
x = mus[s_true] + np.sqrt(sig2s_tmp[s_true]) * np.random.randn(N)
sig2w = np.linalg.norm(np.matmul(A, x)) ** 2 / M * 10 ** (- SNRdB / 10)
w = np.sqrt(sig2w) * np.random.randn(M)
y = np.matmul(A, x) + w

def fbmpr_fxn_reduced(y, A, p1, sig2w, sig2s, mus, D, stop=0):
    M, N = A.shape
    Q = len(mus) - 1
    ps = np.insert(np.ones(Q) * p1 / Q, 0, 1 - p1)
    sig2s = np.hstack([sig2s, sig2s[1] * np.ones(Q - 1)])

    a2 = np.trace(np.matmul(A.T, A)) / N
    nu_true_mean = - M / 2 - M / 2 * np.log(sig2w) - p1 * N / 2 * np.log(a2 * sig2s[1] / sig2w + 1) - M / 2 * np.log(2 * np.pi) + N * np.log(ps[0]) + p1 * N * np.log(ps[1] / ps[0])
    nu_true_stdv = np.sqrt(M / 2 + N * p1 * (1 - p1) * (np.log(ps[1] / ps[0]) - np.log(a2 * sig2s[1] / sig2w + 1) / 2) ** 2)
    nu_stop = nu_true_mean - stop * nu_true_stdv

    psy_thresh = 1e-4
    P = int(min(M, 1 + np.ceil(N * p1 + special.erfcinv(1e-2) * np.sqrt(2 * N * p1 * (1 - p1)))))

    T = [[np.empty(0).astype(int) for i in range(D)] for i in range(P)]
    sT = [[np.empty(0).astype(int) for i in range(D)] for i in range(P)]
    nu = -np.inf * np.ones([P, D])
    xmmse = [[np.empty(0).astype(int) for i in range(D)] for i in range(P)]
    d_tot = np.inf

    nu_root = -np.linalg.norm(y) ** 2 / 2 / sig2w - M * np.log(2 * np.pi) / 2 - M * np.log(sig2w) / 2 + N * np.log(ps[0])
    Bxt_root = A / sig2w
    betaxt_root = abs(sig2s[1] * (1 + sig2s[1] * sum(np.conjugate(A) * Bxt_root)) ** (-1))
    nuxt_root = np.zeros(Q * N)
    for q in range(Q):
        nuxt_root[q * N: N + q * N] = nu_root + np.log(betaxt_root / sig2s[1]) / 2 + 0.5 * betaxt_root * abs(np.matmul(y[None, :], Bxt_root) + mus[q + 1] / sig2s[1]) ** 2 - 0.5 * abs(mus[q + 1]) ** 2 / sig2s[1] + np.log(ps[1] / ps[0])
    
    for d in range(D):
        nuxt = nuxt_root
        z = y
        Bxt = Bxt_root
        betaxt = betaxt_root
        for p in range(P):
            nustar = max(nuxt)
            nqstar = np.argmax(nuxt)
            while sum(abs(nustar - nu[p, :d]) < 1e-8):
                nuxt[nqstar] = -np.inf
                nustar = max(nuxt)
                nqstar = np.argmax(nuxt)
            qstar = int(nqstar // N)
            nstar = int(nqstar % N)
            nu[p, d] = nustar
            T[p][d] = np.append(T[p - 1][d], nstar)
            sT[p][d] = np.append(sT[p - 1][d], qstar)
            z = z - A[:, nstar] * mus[qstar]
            Bxt = Bxt - np.matmul(Bxt[:, nstar][:, None] * betaxt[nstar], np.matmul(Bxt[:, nstar][None, :], A))
            xmmse[p][d] = np.zeros(N)
            xmmse[p][d][T[p][d]] = mus[sT[p][d]] + sig2s[1] * np.matmul(Bxt[:, T[p][d]].T, z[:, None]).flatten()
            betaxt = abs(sig2s[1] * (1 + sig2s[1] * np.sum(np.conjugate(A) * Bxt, axis=0)) ** (-1))
            for q in range(Q):
                nuxt[q * N: q * N + N] = nu[p][d] + np.log(betaxt / sig2s[1]) / 2 + 0.5 * betaxt * abs(np.matmul(z[None, :], Bxt) + mus[q + 1] / sig2s[1]) ** 2 - 0.5 * abs(mus[q + 1]) ** 2 / sig2s[1] + np.log(ps[1] / ps[0])
                nuxt[T[p][d] + q * N] = -np.inf * np.ones(T[p][d].shape)

        if max(nu[:, d]) > nu_stop:
            d_tot = d
            break
    nu = nu[:, :d+1].flatten()

    dum = np.sort(nu)[::-1]
    indx = np.argsort(nu)[::-1]
    d_max = int(np.ceil(indx[0] / P))
    nu_max = nu[indx[0]]
    num = sum(nu > nu_max + np.log(psy_thresh))
    nu_star = nu[indx[:num]]
    psy_star = np.exp(nu_star - nu_max) / sum(np.exp(nu_star - nu_max))
    T_star = [None for i in range(num)]
    xmmse_star = [None for i in range(num)]
    p1_up = 0
    for k in range(num):
        T_star[k] = T[indx[k] % P][indx[k] // P]
        xmmse_star[k] = xmmse[indx[k] % P][indx[k] // P]
        p1_up = p1_up + psy_star[k] * len(T_star[k]) / N

    xmmse = np.matmul(np.vstack([xmmse_star]).T, psy_star[:, None]).flatten()

    return xmmse, xmmse_star, psy_star, nu_star, T_star, d_tot, d_max

xmmse, xmmse_star, psy_star, nu_star, T_star, d_tot, d_max = fbmpr_fxn_reduced(y, A, p1, sig2w, sig2s, mus, D, stop)

NMSE = (np.linalg.norm(xmmse - x) ** 2) / (np.linalg.norm(x) ** 2)

ttl = '||x||_0 = {}'.format(sum(x != 0)) + ', NMSE = {:.2f}'.format(NMSE) + ', N = {}'.format(N) + ', M = {}'.format(M) + ', p_1 = {:.2f}'.format(p1) + ', P_signal = {:.2f}'.format(np.linalg.norm(np.matmul(A, x)) ** 2) + ', P_noise = {:.2f}'.format(np.linalg.norm(w) ** 2)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(211)
ax.vlines(np.arange(N), 0, x, color='r')
ax.vlines(np.arange(N), 0, xmmse, color='b')
ax.vlines(np.arange(N), 0, xmmse_star[0], color='g')
ax.scatter(np.arange(N), x, label='x_true', marker='.', color='r')
ax.scatter(np.arange(N), xmmse, label='x_mmse', marker='+', color='b')
ax.scatter(np.arange(N), xmmse_star[0], label='x_map', marker='x', color='g')
ax.legend()
ax.set_title(ttl)
ax = fig.add_subplot(212)
ax.plot(np.arange(len(psy_star)), psy_star)
ax.set_title('Posterior probabilities of the dominant mixture vectors')
ax.set_xlabel('Top mixture vectors')
ax.set_ylabel('p(s|y)')
plt.savefig('fbmp.png')