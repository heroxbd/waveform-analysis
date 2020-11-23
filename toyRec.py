def deltatime(mu, tau, sigma, N):
    chunk = N // Ncpu + 1
    slices = np.vstack((np.arange(0, N, chunk), np.append(np.arange(chunk, N, chunk), N))).T.astype(np.int).tolist()
    with Pool(min(Ncpu, cpu_count())) as pool:
        result = pool.starmap(partial(start_time, mu=mu, tau=tau, sigma=sigma), slices)

    stime = np.concatenate([result[i][0] for i in range(len(slices))])
    samples = list(itertools.chain.from_iterable([result[i][1] for i in range(len(slices))]))
        
    deltat = stime
    deltat0 = np.array([samples[i][0, 0] for i in range(N)])
    std = np.std(deltat, ddof=-1)
    std0 = np.std(deltat0, ddof=-1)
    return deltat, deltat0, std, std0

def start_time():
    for i in range(a1 - a0):
        logL = lambda t0 : -1 * np.sum(np.log(np.clip(wff.convolve_exp_norm(samples[i][:, 0] - t0, tau, sigma), 
                                                      np.finfo(np.float).tiny, np.inf)))
        if sigma == 0.:
            stime[i] = optimize.minimize(logL, x0=np.min(samples[i][:, 0]), method='SLSQP')['x']
        else:
            stime[i] = optimize.minimize(logL, x0=np.min(samples[i][:, 0])-1, method='L-BFGS-B', bounds=[[-np.inf, samples[i][0, 0]]])['x']

para = {'ideal' : {'tau' : 20., 'sigma' : 0.}, 
        'JNE' : {'tau' : 20., 'sigma' : 1.5}, 
        'JUNO' : {'tau' : 20., 'sigma' : 6.}, 
        'SK' : {'tau' : 0., 'sigma' : 3.}, 
        'slow' : {'tau' : 40., 'sigma' : 2.}, 
        'fast' : {'tau' : 10, 'sigma' : 2.}}

N = int(1e5)
Mu = np.linspace(0.5, 20, num=40)
result = {}

for key in para.keys():
    tau = para[key]['tau']
    sigma = para[key]['sigma']
    Deltaall = np.zeros((len(Mu), N))
    Delta1st = np.zeros((len(Mu), N))
    Stdall = np.zeros(len(Mu))
    Std1st = np.zeros(len(Mu))
    for m in trange(len(Mu), desc=key.ljust(6, ' ')):
        mu = Mu[m]
        Deltaall[m], Delta1st[m], Stdall[m], Std1st[m] = deltatime(mu, tau, sigma, N)
    result.update({key : {'Deltaall' : Deltaall, 'Delta1st' : Delta1st, 'Stdall' : Stdall, 'Std1st' : Std1st}})

fig = plt.figure()
fig.tight_layout()
gs = gridspec.GridSpec(1, 1, figure=fig, left=0.15, right=0.95, top=0.85, bottom=0.1, wspace=0.18, hspace=0.35)

alpha = 0.95

keys = list(para.keys())
ax = fig.add_subplot(gs[0, 0])
for k in range(len(keys)):
    key = keys[k]
    yerr1st = np.vstack([result[key]['Std1st']-np.sqrt(np.power(result[key]['Std1st'],2)*N/chi2.ppf(1-alpha/2, N)), np.sqrt(np.power(result[key]['Std1st'],2)*N/chi2.ppf(alpha/2, N))-result[key]['Std1st']])
    yerrall = np.vstack([result[key]['Stdall']-np.sqrt(np.power(result[key]['Stdall'],2)*N/chi2.ppf(1-alpha/2, N)), np.sqrt(np.power(result[key]['Stdall'],2)*N/chi2.ppf(alpha/2, N))-result[key]['Stdall']])
    ax.errorbar(Mu, result[key]['Std1st'], yerr=yerr1st, label='1st PE', marker='^')
    ax.errorbar(Mu, result[key]['Stdall'], yerr=yerrall, label='all PE', marker='^')
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\delta/\mathrm{ns}$')
    ax.set_title(r'$\tau=${:.01f}'.format(para[key]['tau']) + r'$\mathrm{ns}\ $' + 
                 r'$\sigma=${:.01f}'.format(para[key]['sigma']) + r'$\mathrm{ns}\ $' + 
                 r'$\mathrm{N}=$' + '{0:.1E}\n'.format(N))
    ax.grid()
    if k == 0:
        ax.legend(loc='upper right')
fig.savefig('Note/figures/vs-delta.pgf')
fig.savefig('Note/figures/vs-delta.pdf')
plt.close()
    
fig = plt.figure()
fig.tight_layout()
gs = gridspec.GridSpec(1, 1, figure=fig, left=0.15, right=0.95, top=0.85, bottom=0.1, wspace=0.18, hspace=0.35)

keys = list(para.keys())
ax = fig.add_subplot(gs[0, 0])
for k in range(len(keys)):
    key = keys[k]
    yerr1st = np.vstack([result[key]['Std1st']-np.sqrt(np.power(result[key]['Std1st'],2)*N/chi2.ppf(1-alpha/2, N)), np.sqrt(np.power(result[key]['Std1st'],2)*N/chi2.ppf(alpha/2, N))-result[key]['Std1st']])
    yerrall = np.vstack([result[key]['Stdall']-np.sqrt(np.power(result[key]['Stdall'],2)*N/chi2.ppf(1-alpha/2, N)), np.sqrt(np.power(result[key]['Stdall'],2)*N/chi2.ppf(alpha/2, N))-result[key]['Stdall']])
    yerr = np.vstack([result[key]['Stdall'] / result[key]['Std1st'] - (result[key]['Stdall'] - yerrall[0]) / (result[key]['Std1st'] + yerr1st[1]), 
                      (result[key]['Stdall'] + yerrall[1]) / (result[key]['Std1st'] - yerr1st[0]) - result[key]['Stdall'] / result[key]['Std1st']])
    ax.errorbar(Mu, result[key]['Stdall'] / result[key]['Std1st'], yerr=yerr, label=r'$\frac{\delta_{1st}}{\delta_{all}}$', marker='^')
    ax.set_ylim(0.9, 1.01)
    ax.hlines(1, xmin=Mu[0], xmax=Mu[-1], color='k')
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\delta_{all}-\delta_{1st}/\mathrm{ns}$')
    ax.set_title(r'$\tau=${:.01f}'.format(para[key]['tau']) + r'$\mathrm{ns}\ $' + 
                 r'$\sigma=${:.01f}'.format(para[key]['sigma']) + r'$\mathrm{ns}\ $' + 
                 r'$\mathrm{N}=$' + '{0:.1E}\n'.format(N))
    ax.grid()
    if k == 0:
        ax.legend(loc='lower right')
fig.savefig('Note/figures/vs-deltasub.pgf')
fig.savefig('Note/figures/vs-deltasub.pdf')
plt.close()