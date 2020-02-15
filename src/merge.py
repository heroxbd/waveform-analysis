import h5py
import argparse
import numpy as np
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-o', dest='opt', help='output')
    psr.add_argument('ipt', nargs='+', help='input')
    args = psr.parse_args()
    with h5py.File(args.opt,'w') as opt:
        opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint8), ('PEnum', np.uint16), ('wdist', np.float32), ('pdist', np.float32)])
        length = len(args.ipt)
        N = 17613
        dt = np.zeros(N*length, dtype=opdt)
        for i in range(length):
            name = args.ipt[i]
            with h5py.File(name, 'r') as ipt:
                print('append {}'.format(name))
                temp = ipt['Record'][:]
                temp['EventID'] = int(name.split('/wa')[1].split('ve')[1].split('/')[0])
                dt[i*N:(i+1)*N] = temp[0:N]
        opt.create_dataset('Record', data=dt, compression='gzip')