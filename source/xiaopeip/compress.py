import h5py
import numpy as np
with h5py.File("finalanswer.h5") as ipt:
    data=ipt["Answer"][()]

with h5py.File("compressedanswer.h5","w") as opt:
    opt.create_dataset("Answer",data=data,compression="gzip", shuffle=True)
