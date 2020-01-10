import sys
import h5py
from scipy.signal import convolve
import tables
import numpy as np
class AnswerData(tables.IsDescription):
    EventID=tables.Int64Col(pos=0)
    ChannelID=tables.Int16Col(pos=1)
    PETime=tables.Int16Col(pos=2)
    Weight=tables.Float32Col(pos=3)
def ReadWave(filename):
    h5file=tables.open_file(filename,"r")
    waveTable=h5file.root.Waveform
    entry=0
    wave=waveTable[:]['Waveform']
    eventId=waveTable[:]['EventID']
    channelId=waveTable[:]['ChannelID']
    h5file.close()
    return (wave,eventId,channelId)
def lucyDDM_N(waveform, spe):
    signal = np.zeros(waveform.shape)
    length = waveform.shape[0]
    moveDelta = 10
    spe = np.append(np.zeros((spe.shape[0]- 2*moveDelta + 1,)), np.abs(spe))
    for i in range(length):
        signal[i, :] = lucyDDM(waveform[i, :], spe, 100)
        print('\rThe analysis processing:|{}>{}|{:6.2f}%'.format(((20*i)//length)*'-', (19 - (20*i)//length)*' ', 100 * ((i+1) / length)), end=''if i != length-1 else '\n') # show process bar
    return signal
def lucyDDM(waveform, spe, iterations=50):
    '''Lucy deconvolution
    Parameters
    ----------
    waveform : 1d array
    spe : 1d array
        point spread function; single photon electron response
    iterations : int

    Returns
    -------
    signal : 1d array
    '''
    # abs waveform, spe
    waveform = np.abs(waveform)
    # spe = np.append(np.zeros((spe.shape[0]-1,)), np.abs(spe))
    # use the deconvlution method
    wave_deconv = np.full(waveform.shape, 0.5)
    spe_mirror = spe[::-1]
    for _ in range(iterations):
        relative_blur = waveform / convolve(wave_deconv, spe, mode='same')
        wave_deconv *= convolve(relative_blur, spe_mirror, mode='same')
        # there is no need to set the bound if the spe and the wave are all none negative 
    return wave_deconv
def writeSubfile(truth, eventId, channelId, sigma, moveDelta, subfile):
    answerh5file = tables.open_file(subfile, mode='w', title="OneTonDetector")
    AnswerTable = answerh5file.create_table('/', 'Answer', AnswerData, 'Answer')
    answer = AnswerTable.row
    rangelist = np.arange(truth.shape[1])
    length = len(eventId)
    for i in range(length):
        eventIndex = eventId[i]
        channelIndex = channelId[i]
        waitWriteTruth = rangelist[truth[i, :] > (sigma * np.std(truth[i, :]))]
        for t in waitWriteTruth:# wait for the null output process
            if truth[i, t] > np.std(truth[i, (t-5):(t+5)]):
                answer['EventID'] = eventIndex
                answer['ChannelID'] = channelIndex
                answer['PETime'] = t - moveDelta
                answer['Weight'] = truth[i, t]           
                answer.append()
        print('\rThe writing processing:|{}>{}|{:6.2f}%'.format(((20*i)//length)*'-', (19 - (20*i)//length)*' ', 100 * ((i+1) / length)), end=''if i != length-1 else '\n') # show process bar
    AnswerTable.flush()
    answerh5file.close()
if __name__ == "__main__":
    if len(sys.argv)<2:
        problemfile = './data/simulate/hdf5/ftraining-0.h5'
        subfile = './data/analysis/LucyDDM_python/ftraining-0/ftraining-0Answer.h5'
        spe = np.load('analysis/LucyDDM_python/spe.npy')
    else:
        problemfile = sys.argv[1]
        subfile = sys.argv[3]
        # spe = np.load(sys.argv[2])
        with h5py.File(sys.argv[2]) as ipt:
            spe = ipt['spe'][:]
    spePart = spe[0:40].reshape((40,))
    if spePart[0]==0:
        spePart[0] = np.abs(spePart[1])
    (waveform, eventId, channelId) = ReadWave(problemfile)
    numPMT = np.max(channelId)
    length = waveform.shape[1]
    waveformNobase = (np.sum(waveform[:, 0:150], axis=1)/150).reshape((waveform.shape[0], 1)).repeat(length, axis=1)- waveform
    if np.min(waveformNobase[0,:])<-10:
        waveformNobase = -waveformNobase
    print('Begin analyze {}'.format(problemfile))
    lucyTruth = lucyDDM_N(waveformNobase, spePart)
    writeSubfile(lucyTruth, eventId, channelId, 7, 9, subfile)
    print('End write {}'.format(subfile))

