import math

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import scipy.special as special

import wf_func as wff

import matplotlib
matplotlib.use('pgf')
plt.style.use('default')

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

xmmse, xmmse_star, psy_star, nu_star, T_star, d_tot, d_max = wff.fbmpr_fxn_reduced(y, A, p1, sig2w, sig2s, mus, D, stop)

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