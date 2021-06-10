import numpy as np
import numpyro
import wf_func as wff

from scipy import optimize, special
from scipy.stats import poisson
templateProbability = np.array([68.74,16.58,8.595,6.022])
normalTplProbability = templateProbability/np.sum(templateProbability)
defaultHnumax = 200
defaultpetimes = 4
# calculate the probability matrxi
def hnu2PEProbability(probability,hnumax=200,petimes=4):
    # probability is the mcp charge response fit result, the first row and first column is for zero hnu and PE
    E = np.zeros((hnumax+1, hnumax*petimes+1))
    E[0,0] = 1
    E[1,1:(petimes+1)] = probability
    # hnumax+1 is the hnumax hnu
    for i in range(2,hnumax+1):
        for j in range(i,5):
            tmpE = np.zeros(4)
            tmpE[:j] = E[(i-1),(j-1)::-1]
            E[i,j] = np.sum(tmpE*probability)
        for j in range(5, 4*i+1):
            tmpE = E[(i-1),(j-1):(j-5):-1]
            E[i,j] = np.sum(tmpE*probability)
    return E
hnu2PEmatrix = hnu2PEProbability(normalTplProbability)
def lightCurve(t0, mu, binsT, tau, sigma, binwidth=0.1):
    # use the ligth curve distribution from wff
    expectM = mu * wff.convolve_exp_norm(binsT-t0, tau, sigma)*binwidth
    return expectM
def hypo0(expectM):
    return np.exp(-expectM)
def hypo1(expecthnu, hnu2PEp):
    return np.dot(expecthnu, hnu2PEp)
def hypo(expectM, cstar, indexstar):
    binsp = np.tile(hypo0(expectM), (cstar.shape[0],1))
    for i in range(cstar.shape[0]):
        for j in range(len(indexstar[i])):
            # indexstar[i][j]  is the index of nonzeros
            expecthnu = poisson.pmf(np.arange(defaultHnumax+1), expectM[indexstar[i][j]])
            assert(cstar[i,indexstar[i][j]]>=1)
            binsp[i, indexstar[i][j]] = hypo1(expecthnu, hnu2PEmatrix[:,cstar[i,indexstar[i][j]]])
    return binsp
def likelihood(x,*args):
    binsT, tau, sigma,c_star, p_star, index_star, mu = args
    lightC = lightCurve(x[0],mu,binsT,tau,sigma)
    prob = hypo(lightC, c_star, index_star)
    L = -special.logsumexp(np.sum(np.log(prob),axis=1),b=p_star)
    #L = -np.dot(np.sum(np.log(prob),axis=1),p_star)
    return L
def findNonzero(cstar):
    indexstar = [[]]*cstar.shape[0]
    for i in range(cstar.shape[0]):
        indexstar[i] = np.where(cstar[i]>0)[0]
    return indexstar
def optimizet0mu(binsT, tau, sigma,c_star, p_star, t0guess,method='SLSQP'):
    index_star = findNonzero(c_star)
    expectnum = np.sum(c_star)/len(c_star)
    fitresults = []
    for delta in np.arange(0,100,0.2):
        for n in range(1,int(expectnum)+2):
            x0 = [t0guess-delta]
            fitresults.append((optimize.minimize(likelihood, x0, method=method,bounds=((t0guess-200, t0guess+50),(0,500)),args=(binsT, tau, sigma,c_star, p_star, index_star,n)),n))#,options={'eps':0.01}))
    return min(fitresults,key=lambda x:x[0].fun)
def optimizeBotht0mu(binsT, tau, sigma,c_star, p_star, t0guess,method='SLSQP'):
    # minimize t0 and mu both
    index_star = findNonzero(c_star)
    expectnum = np.sum(c_star)/len(c_star)
    fitresults = []
    for delta in np.arange(0,100,0.2):
        for n in range(1,int(expectnum)+2):
            x0 = [t0guess-delta, n]
            fitresults.append(optimize.minimize(likelihood, x0, method=method,bounds=((t0guess-200, t0guess+50),(0,500)),args=(binsT, tau, sigma,c_star, p_star, index_star,n)))#,options={'eps':0.01}))
    return min(fitresults,key=lambda x:x.fun)
# define the process from number of photon to pe
def photon2pe(nphoton, pmf=normalTplProbability):
    return np.random.choice(np.arange(1,templateProbability.shape[0]+1),size=nphoton, p=pmf)
#class mcpModel(numpyro.distributions.distribution.Distribution):

def mcpModel(pes, pmf=normalTplProbability,logits=range(1,len(normalTplProbability)+1)):
    # pe expect value
    p0 = numpyro.sample('p0', numpyro.distributions.Uniform(0,10))
    # pe poisson value
    p1 = numpyro.sample('p1', numpyro.distributions.Poisson(p0))
    obs = 0
    for i in range(p1):
        obs += numpyro.sample('obs', numpyro.distributions.Categorical(pmf, logits))
    return obs
def mcpAllocateCharge(ps, charges):
    pcs = np.zeros(ps.shape)
    begin = 0
    for i in range(len(ps)):
        pcs[i] = np.sum(charges[begin:(begin+ps[i])])
    return pcs
def mcpMC(pes):
    nuts_kernel = numpyro.infer.NUTS(mcpModel, init_strategy=numpyro.infer.initialization.init_to_value(values={'p0': jnp.array([0.5])}))
    mcmc = numpyro.infer.MCMC(nuts_kernel, num_samples=100,num_warmup=30,progress_bar=False)
    mcmc.run(rng_key,pes=pes)
    p0 = np.array(mcmc.get_samples()['p0'])
    p0 = np.array(mcmc.get_samples()['p1'])
def fbmp_multi_gaussian(y, A, p1, sig2w, sig2s, mus, D, stop=0):
    # Only for multi-gaussian with arithmetic sequence of mu and sigma
    M, N = A.shape

    p = 1 - poisson.pmf(0, p1).mean()
    nu_true_mean = -M / 2 - M / 2 * np.log(sig2w) - p * N / 2 * np.log(sig2s / sig2w + 1) - M / 2 * np.log(2 * np.pi) + N * np.log(1 - p) + p * N * np.log(p / (1 - p))
    nu_true_stdv = np.sqrt(M / 2 + N * p * (1 - p) * (np.log(p / (1 - p)) - np.log(sig2s / sig2w + 1) / 2) ** 2)
    nu_stop = nu_true_mean + stop * nu_true_stdv

    psy_thresh = 1e-3
    P = min(M, 1 + math.ceil(N * p + special.erfcinv(1e-2) * math.sqrt(2 * N * p * (1 - p))))
    # P = math.ceil(min(M, p1.sum() + 3 * np.sqrt(p1.sum())))
    D = min(min(len(p1), P), D)

    T = np.full((P, D), 0)
    nu = np.full((P, D), -np.inf)
    xmmse = np.zeros((P, D, N))
    cc = np.zeros((P, D, N))
    d_tot = D

    nu_root = -0.5 * np.linalg.norm(y) ** 2 / sig2w - 0.5 * M * np.log(2 * np.pi) - 0.5 * M * np.log(sig2w) + np.log(poisson.pmf(0, p1)).sum()
    cx_root = A / sig2w
    betaxt_root = np.abs(sig2s / (1 + sig2s * np.sum(A * cx_root, axis=0)))
    nuxt_root = nu_root + 0.5 * np.log(betaxt_root / sig2s) + 0.5 * betaxt_root * np.abs(np.dot(y, cx_root) + mus / sig2s) ** 2 - 0.5 * mus ** 2 / sig2s + np.log(poisson.pmf(1, p1) / poisson.pmf(0, p1))
    
    for d in range(D):
        nuxt = nuxt_root.copy()
        z = y.copy()
        cx = cx_root.copy()
        betaxt = betaxt_root.copy()
        for p in range(P):
            nuxtshadow = np.where(np.sum(np.abs(nuxt - nu[p, :d][:, None]) < 1e-4, axis=0), -np.inf, nuxt)
            nustar = max(nuxtshadow)
            istar = np.argmax(nuxtshadow)
            nu[p, d] = nustar
            T[p, d] = istar
            cx = cx - np.dot(betaxt[istar] * cx[:, istar].copy().reshape(M, 1), np.dot(cx[:, istar], A).copy().reshape(1, N))

            z = z - A[:, istar] * mus
            assist = np.zeros(N)
            t, c = np.unique(T[:p+1, d], return_counts=True)
            assist[t] = mus * c + sig2s * c * np.dot(z, cx[:, t])
            cc[p, d][t] = c
            xmmse[p, d] = assist
            # poisson
            betaxt = np.abs(sig2s / (1 + sig2s * np.sum(A * cx, axis=0)))
            nuxt = nustar + 0.5 * np.log(betaxt / sig2s) + 0.5 * betaxt * np.abs(np.dot(z, cx) + mus / sig2s) ** 2 - 0.5 * mus ** 2 / sig2s + np.log(poisson.pmf(np.sum(T[:p, d] == istar) + 1, p1[istar]) / poisson.pmf(np.sum(T[:p, d] == istar), p1[istar]))

        if max(nu[:, d]) > nu_stop:
            d_tot = d + 1
            break
    # revise = np.log(poisson.pmf(np.arange(1, P+1), p1.sum())) - special.logsumexp(np.log(poisson.pmf(cc[:, :d_tot], p1)).sum(axis=-1), axis=1)
    # revise = np.log(poisson.pmf(np.arange(1, P+1), p1.sum()))
    # nu[:, :d_tot] = nu[:, :d_tot] + revise[:, None]
    # pp = poisson.pmf(np.arange(1, P+1), p1.sum())
    # nu[:, :d_tot][np.random.uniform(size=nu[:, :d_tot].shape) > pp[:, None] / pp.max()] = -np.inf
    nu = nu[:, :d_tot].T.flatten()

    dum = np.sort(nu)[::-1]
    indx = np.argsort(nu)[::-1]
    d_max = math.floor(indx[0] // P) + 1
    num = min(int(np.sum(nu > nu.max() + np.log(psy_thresh))), D)
    nu_star = nu[indx[:num]]
    psy_star = np.exp(nu_star - nu.max()) / np.sum(np.exp(nu_star - nu.max()))
    T_star = [T[:(indx[k] % P) + 1, indx[k] // P] for k in range(num)]
    xmmse_star = np.empty((num, N))
    for k in range(num):
        xmmse_star[k] = xmmse[indx[k] % P, indx[k] // P]

    xmmse = np.average(xmmse_star, weights=psy_star, axis=0)

    return xmmse, xmmse_star, psy_star, nu_star, T_star, d_tot, d_max
