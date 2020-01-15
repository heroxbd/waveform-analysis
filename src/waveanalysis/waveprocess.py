import sys
import h5py
from scipy.signal import convolve
import tables
import numpy as np

from scipy.fftpack import fft, ifft
class AnswerData(tables.IsDescription):
    EventID=tables.Int64Col(pos=0)
    ChannelID=tables.Int16Col(pos=1)
    PETime=tables.Int16Col(pos=2)
    Weight=tables.Float32Col(pos=3)
def lucyDDM_N(waveforms, spe):
    signal = np.zeros(waveforms.shape)
    length = waveforms.shape[0]
    moveDelta = 10
    spe = np.append(np.zeros((spe.shape[0]- 2*moveDelta + 1,)), np.abs(spe))
    for i in range(length):
        signal[i, :] = lucyDDM(waveforms[i, :], spe, 100)
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
    
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    .. [2] https://github.com/scikit-image/scikit-image/blob/master/skimage/restoration/deconvolution.py#L329
    '''
    # abs waveform, spe
    waveform = np.abs(waveform)
    waveform = waveform/np.sum(spe)
    # spe = np.append(np.zeros((spe.shape[0]-1,)), np.abs(spe))
    # use the deconvlution method
    wave_deconv = np.full(waveform.shape, 0.5)
    spe_mirror = spe[::-1]
    for _ in range(iterations):
        relative_blur = waveform / convolve(wave_deconv, spe, mode='same')
        wave_deconv *= convolve(relative_blur, spe_mirror, mode='same')
        # there is no need to set the bound if the spe and the wave are all none negative 
    return wave_deconv
def writeSubfile(truth, eventId, channelId, sigma, moveDelta, subfile, mode='lack'):
    answerh5file = tables.open_file(subfile, mode='w', title="OneTonDetector")
    AnswerTable = answerh5file.create_table('/', 'Answer', AnswerData, 'Answer')
    answer = AnswerTable.row
    rangelist = np.arange(truth.shape[1])
    length = len(eventId)
    for i in range(length):
        eventIndex = eventId[i]
        channelIndex = channelId[i]
        waitWriteTruth = rangelist[truth[i, :] > (sigma * np.std(truth[i, :]))]
        counter = 0
        for t in waitWriteTruth:# wait for the null output process
            if truth[i, t] > np.std(truth[i, (t-5):(t+5)]):
                answer['EventID'] = eventIndex
                answer['ChannelID'] = channelIndex
                answer['PETime'] = t - moveDelta
                answer['Weight'] = truth[i, t]           
                answer.append()
                counter += 1
        if counter == 0 and mode == 'full':
            answer['EventID'] = eventIndex
            answer['ChannelID'] = channelIndex
            answer['PETime'] = 300
            answer['Weight'] = 1           
            answer.append()    
        print('\rThe writing processing:|{}>{}|{:6.2f}%'.format(((20*i)//length)*'-', (19 - (20*i)//length)*' ', 100 * ((i+1) / length)), end=''if i != length-1 else '\n') # show process bar
    AnswerTable.flush()
    answerh5file.close()
def xdcFTspe(spe, length, AXE=4, EXP=4):
    stdmodel = np.where(stdmodel>0.02, spe, 0)
    model = fft(stdmodel)
    model = np.where(model > AXE, model - AXE, 0)
    core = model / np.max(model)
    for i in range(len(core)):
        core[i] = pow(core[i], EXP)
    model = core * np.max(model)
    model = np.where(model > 0.02, model, 0)
    model_raw = np.concatenate([model, np.zeros(length - len(model))])
    model_k = fft(model_raw)
    return model_k
def xdcFT(waveform, spefft, KNIFE=0.05, AXE=4):
    wf_input = np.mean(waveform[-101:-1]) - waveform
    wf_input = np.where(wf_input > 0, wf_input, 0)
    wf_input = np.where(wf_input > AXE, wf_input - AXE, 0)
    wf_k = fft(wf_input)
    spec = np.divide(wf_k, spefft)
    pf = ifft(spec)
    pf = pf.real
    pf = np.where(pf >KNIFE, pf, 0)
    lenpf = np.size(np.where(pf>0))
    if lenpf ==0:
        pf[300] = 1
        lenpf = 1
    pet = np.where(pf>0)[0]
    pwe = pf[pf > 0]
    pwe = pwe.astype(np.float16)
    return pet, pwe
def xdcFTN(waveforms, spe, eventID, channelID):
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint8), ('PETime', np.uint16), ('Weight', np.float16)])
    length = waveforms.shape[0]
    start = 0
    dt = np.zeros(50 * length, dtype=opdt)
    spefft =xdcFTspe(spe, waveforms.shape[1])
    for i in range(length):
        pet, pwe = xdcFT(waveforms[i], spefft)
        end = start + len(pet)
        dt['PETime'][start:end] = pet
        dt['Weight'][start:end] = pwe
        dt['EventID'][start:end] = eventID[i]
        dt['ChannelID'][start:end] = channelID[i]
        start = end
        print('\rThe analysis processing:|{}>{}|{:6.2f}%'.format(((20*i)//length)*'-', (19 - (20*i)//length)*' ', 100 * ((i+1) / length)), end=''if i != length-1 else '\n') # show process bar
    return dt[0:end]

