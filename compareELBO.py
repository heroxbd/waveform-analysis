#!/usr/bin/python3
import h5py, numpy as np
import argparse
import matplotlib.pyplot as plt
'''
Compare the ELBO between xdc and zaq
'''
psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', type=str, help='output file')
psr.add_argument('--tau', nargs='+', help='input tau')
psr.add_argument('--sigma', nargs='+', help='input sigma')
psr.add_argument('--type', nargs='+', help='prior type')
psr.add_argument('--idir', nargs='+', help='input directory')
psr.add_argument('--describe', nargs='+', help='description')
psr.add_argument('--mus', nargs='+', help='mus of scan')
psr.add_argument('--field', default='elbo', help='the field of the dataset')
args = psr.parse_args()

elbos = np.zeros((len(args.idir)*len(args.type),len(args.tau), len(args.sigma), len(args.mus)))
elboSigmas = np.zeros((len(args.idir)*len(args.type),len(args.tau), len(args.sigma), len(args.mus)))
typelen = len(args.type)
mus = [float(mu) for mu in args.mus]
for i,id in enumerate(args.idir):
    for j,t in enumerate(args.type):
        for k,ta in enumerate(args.tau):
            for l,s in enumerate(args.sigma):
                for m,mu in enumerate(mus):
                    if args.field =='elbo':
                        with h5py.File(id+'/'+t+'/char/'+'{:.1f}-{}-{}.h5'.format(mu,ta,s),'r') as ipt:
                            elbo = ipt[args.field][:]
                    elif args.field =='wdist':
                        with h5py.File(id+'/'+t+'/dist/'+'{:.1f}-{}-{}.h5'.format(mu,ta,s),'r') as ipt:
                            elbo = ipt['Record'][args.field][:]
                    elbos[i*typelen+j,k,l,m] = np.average(elbo)
                    elboSigmas[i*typelen+j,k,l,m] = np.std(elbo)
fig, axs = plt.subplots(1,2,figsize=(16,6))

for i,id in enumerate(args.describe):
    for j,t in enumerate(args.type):
        for k,ta in enumerate(args.tau):
            for l,s in enumerate(args.sigma):
                axs[0].plot(mus, elbos[i*typelen+j,k,l], label=id+'{}-{}-{}'.format(t,ta,s))
                axs[1].plot(mus, elboSigmas[i*typelen+j,k,l], label=id+'{}-{}-{}'.format(t,ta,s))
axs[0].set_title('$\mu('+args.field+')-\mu_{PE}$')
axs[1].set_title('$\sigma('+args.field+')-\mu_{PE}$')
axs[0].set_xlabel('$\mu_{PE}$')
axs[1].set_xlabel('$\mu_{PE}$')
axs[0].legend()
axs[1].legend()

plt.tight_layout()
plt.savefig(args.opt)