import numpy as np, h5py
fipt = "answerfile.h5"
fopt = "submissionfile.h5"

opd = [('EventID', '<i8'), ('ChannelID', '<i2'),
       ('PETime', 'f4'), ('Weight', 'f4')]

with h5py.File(fipt) as ipt, h5py.File(fopt, "w") as opt: #将matlab生成的answerfile.h5文件（未压缩且格式不符合提交要求）转化为可供提交的h5文件
        rst = np.zeros(len(ipt['PETime'][0]), dtype=opd)
        rst['PETime'] = ipt['PETime'][0]
        rst['Weight'] = (ipt['Weight'][0])
        rst['EventID'] = ipt['EventID'][0]
        rst['ChannelID'] = ipt['ChannelID'][0]
        opt.create_dataset('Answer', data=rst, compression='gzip')
        