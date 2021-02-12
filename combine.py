import os
import re

import h5py
import numpy as np
from tqdm import tqdm

def combinef(path, ipt, ipt_1):
    if path in ipt and path in ipt_1:
        D = ipt[path][:]
        D_1 = ipt_1[path][:]
        del ipt[path]
        ipt.create_dataset(path, data=np.append(D, D_1), compression='gzip', compression_opts=4)
    elif path in ipt and not path in ipt_1:
        pass
    elif not path in ipt and path in ipt_1:
        D_1 = ipt_1[path][:]
        ipt.create_dataset(path, data=D_1, compression='gzip', compression_opts=4)
    return

Filelist = os.listdir('.')
Filelist.sort()
filelist = []
pat = re.compile(r'1t_\+0\.[0-9]{3}_[x,z]\.h5')
for foo in Filelist:
    if re.search(pat, foo) is not None and foo[:-3] + '_1.h5' in Filelist:
        filelist.append(foo)

for foo in tqdm(filelist):
    foo_1 = foo[:-3] + '_1.h5'
    with h5py.File(foo, 'r+', libver='latest', swmr=True) as ipt, h5py.File(foo_1, 'r', libver='latest', swmr=True) as ipt_1:
        paths = ['RunHeader', 'SimTriggerInfo/PEList', 'SimTriggerInfo/TruthList', 'SimTruth/SimTruth', 'SimTruth/DepositEnergy', 'SimTruth/PrimaryParticle', 'Readout/TriggerInfo', 'Readout/Waveform']
        for path in paths:
            combinef(path, ipt, ipt_1)
    # os.remove(foo_1)