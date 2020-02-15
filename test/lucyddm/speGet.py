import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import tables
import itertools as it
def speTruth(waveform, truth, spe):
    # use the small signal which may be spe signal to get spe
    speSum = np.array(spe)
    number = 0
    truth = np.append(np.sort(truth), 1000)
    t = np.zeros(truth.shape, dtype=int)
    waveform = waveform - np.mean(waveform[-100:-1])
    if np.mean(waveform)<0:
        waveform = -waveform
    for i in range(1, len(truth)-1):
        if (truth[i]-truth[i-1]) > 50 and (truth[i+1]-truth[i]) > 50:
            tempWaveform = waveform[int(truth[i]):(int(truth[i])+50)]-waveform[int(truth[i])]
            if np.mean(tempWaveform[5:15])>10:
                speSum += tempWaveform
                t[number] = int(truth[i])
                number += 1
    if number ==0:
        return spe, [], 0
    else:
        return speSum/number, t[0:number], number
def writeSpeH5(spe, filename="spe.h5"):
    with h5py.File(filename, 'w') as opt:
        opt.create_dataset('spe',data=spe,compression='gzip')

def ReadWavejp(filename):
    h5file = tables.open_file(filename, "r")
    waveTable = h5file.root.Waveform
    waveform = waveTable[:]['Waveform']
    eventID = waveTable[:]['EventID']
    channelID = waveTable[:]['ChannelID']
    truthTable = h5file.root.GroundTruth
    truth = truthTable[:]
    h5file.close()
    return (waveform, eventID, channelID, truth)
if __name__ == "__main__":
    if len(sys.argv)<2:
        outFile = 'output/jinping/lucyddm'
        trainWaveFile = 'ztraining-0.h5'
    else:
        outFile = sys.argv[2]
        trainWaveFile = sys.argv[1]
    delta = 0
    spelength = 50
    trainMax = 1000
    num = 0
    speZeros = np.zeros((1,spelength)).reshape(spelength,)
    speWaveSum = np.zeros((1,spelength)).reshape(spelength,)
    (waveforms, eid , ch, truth) =  ReadWavejp(trainWaveFile)
    numPmt = np.max(ch)
    v_truth = truth['EventID'] * numPmt + truth['ChannelID']
    v_truth, i_truth = np.unique(v_truth, return_index=True)
    print("waveform shape is {}".format(waveforms.shape))
    waveNumber = waveforms.shape[0]
    rowid = 0
    n = 0
    for v, i0, i in zip(v_truth, np.nditer(i_truth), it.chain(np.nditer(i_truth[1:]), [len(truth['EventID'])])):
        eventid = truth[i0]['EventID']
        channelid = truth[i0]['ChannelID']
        if eid[rowid] != eventid:
            break
        if ch[rowid] == channelid:
            w_truth = truth[i0:i]['PETime'] + delta
            (speWave, t, n) = speTruth(waveforms[rowid], w_truth, speZeros)
            # speGet.plotSpe(waveforms[rowid], speWave, t, n)
            rowid += 1
        elif ch[rowid] <channelid:
            rowid += 1
            continue
        elif ch[rowid] > channelid:
            continue
        if n>0:
            num += 1
            speWaveSum += speWave
        if num > trainMax:
            break
        print('\rThe Single PE Generating:|{}>{}|{:6.2f}%'.format(((20*num)//trainMax)*'-', (20 - (20*num)//trainMax)*' ', 100 * ((num) / trainMax)), end=''if num != trainMax else '\n') # show process bar
    print("eventid {},channelid {}, rowid {}".format(eventid, channelid, rowid))
    if num == 0:
        print("warning, nospe")
    else:
        # np.save('./spe.npy', speWaveSum/num)
        writeSpeH5(speWaveSum/num, outFile)