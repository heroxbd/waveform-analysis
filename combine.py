# -*- coding: utf-8 -*-

import os
import re

import h5py
import numpy as np
from tqdm import tqdm

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
        RunHeader = ipt['RunHeader'][:]
        RunHeader_1 = ipt_1['RunHeader'][:]
        del ipt['RunHeader']
        ipt.create_dataset('RunHeader', data=np.append(RunHeader, RunHeader_1), compression='gzip', compression_opts=4)
        PEList = ipt['SimTriggerInfo/PEList'][:]
        PEList_1 = ipt_1['SimTriggerInfo/PEList'][:]
        del ipt['SimTriggerInfo/PEList']
        ipt.create_dataset('SimTriggerInfo/PEList', data=np.append(PEList, PEList_1), compression='gzip', compression_opts=4)
        TruthList = ipt['SimTriggerInfo/TruthList'][:]
        TruthList_1 = ipt_1['SimTriggerInfo/TruthList'][:]
        del ipt['SimTriggerInfo/TruthList']
        ipt.create_dataset('SimTriggerInfo/TruthList', data=np.append(TruthList, TruthList_1), compression='gzip', compression_opts=4)
        SimTruth = ipt['SimTruth/SimTruth'][:]
        SimTruth_1 = ipt_1['SimTruth/SimTruth'][:]
        del ipt['SimTruth/SimTruth']
        ipt.create_dataset('SimTruth/SimTruth', data=np.append(SimTruth, SimTruth_1), compression='gzip', compression_opts=4)
        DepositEnergy = ipt['SimTruth/DepositEnergy'][:]
        DepositEnergy_1 = ipt_1['SimTruth/DepositEnergy'][:]
        del ipt['SimTruth/DepositEnergy']
        ipt.create_dataset('SimTruth/DepositEnergy', data=np.append(DepositEnergy, DepositEnergy_1), compression='gzip', compression_opts=4)
        PrimaryParticle = ipt['SimTruth/PrimaryParticle'][:]
        PrimaryParticle_1 = ipt_1['SimTruth/PrimaryParticle'][:]
        del ipt['SimTruth/PrimaryParticle']
        ipt.create_dataset('SimTruth/PrimaryParticle', data=np.append(PrimaryParticle, PrimaryParticle_1), compression='gzip', compression_opts=4)
        TriggerInfo = ipt['Readout/TriggerInfo'][:]
        TriggerInfo_1 = ipt_1['Readout/TriggerInfo'][:]
        del ipt['Readout/TriggerInfo']
        ipt.create_dataset('Readout/TriggerInfo', data=np.append(TriggerInfo, TriggerInfo_1), compression='gzip', compression_opts=4)
        Waveform = ipt['Readout/Waveform'][:]
        Waveform_1 = ipt_1['Readout/Waveform'][:]
        del ipt['Readout/Waveform']
        ipt.create_dataset('Readout/Waveform', data=np.append(Waveform, Waveform_1), compression='gzip', compression_opts=4)
    os.remove(foo_1)