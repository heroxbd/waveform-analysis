import sys
import h5py
from scipy.signal import convolve
import numpy as np
import tables
from scipy.fftpack import fft, ifft
class AnswerData(tables.IsDescription):
    EventID=tables.Int64Col(pos=0)
    ChannelID=tables.Int16Col(pos=1)
    PETime=tables.Int16Col(pos=2)
    Weight=tables.Float32Col(pos=3)

def recon_N(waveform, eventId, channelId, spe, method='lucyddm', negativePulse=True):
    if negativePulse:
        waveformNobase = np.mean(waveform[:,0:100], axis=1).reshape((waveform.shape[0],1)) - waveform
    else:
        waveformNobase = waveform-np.mean(waveform[:,0:100], axis=1).reshape((waveform.shape[0],1))
    if method=='lucyddm':
        return lucyDDM_N(waveformNobase, spe)
    elif method == 'threshold':
        return threshold_N(waveformNobase, eventId, channelId, spe)
    elif method == 'xdcft':
        return xdcFT_N(waveformNobase, eventId, channelId, spe)
    elif method == 'fft':
        return fft_N(waveformNobase, eventId, channelId, spe)
    else:
        return waveform
def fft_N(waveforms, eventID, channelID, spe):
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    length = waveforms.shape[0]
    start = 0
    dt = np.zeros(100 * length, dtype=opdt)
    spefft = fft(spe, 2* waveforms.shape[1])
    for i in range(length):
        pet, pwe = fft_aq(waveforms[i], spefft)
        if len(pet) == 0:
            dt['PETime'][start:end] = 300
            dt['Weight'][start:end] = 1
            dt['EventID'][start:end] = eventID[i]
            dt['ChannelID'][start:end] = channelID[i]
            start = start +1
        else:
            end = start + len(pet)
            dt['PETime'][start:end] = pet
            dt['Weight'][start:end] = pwe
            dt['EventID'][start:end] = eventID[i]
            dt['ChannelID'][start:end] = channelID[i]
            start = end
        print('\rThe analysis processing:|{}>{}|{:6.2f}%'.format(((20*i)//length)*'-', (19 - (20*i)//length)*' ', 100 * ((i+1) / length)), end=''if i != length-1 else '\n') # show process bar
    return dt[0:start]
def fft_aq(waveform, spefft, sigma=3):
    length = waveform.shape[0]
    wavef = fft(waveform, 2* length)
    wavef[(length-int(length*0.75)):(length+int(length*0.75))] = 0
    signalf = np.true_divide(wavef, spefft)
    signal = np.real(ifft(signalf,2*length))
    signal[signal<0]=0
    basethreshold = np.mean(signal[0:100]) + sigma * np.std(signal)
    recon = np.where(signal>basethreshold)[0]
    weight = signal[recon]
    return recon, weight
def xdcFTspe(spe, length, AXE=4, EXP=4):
    stdmodel = np.where(spe>0.02, spe, 0)
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
def xdcFT_N(waveforms, eventID, channelID, spe):
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float16)])
    length = waveforms.shape[0]
    start = 0
    dt = np.zeros(50 * length, dtype=opdt)
    spefft =xdcFTspe(spe, waveforms.shape[1])
    for i in range(length):
        pet, pwe = xdcFT(waveforms[i], spefft)
        if len(pet) == 0:
            dt['PETime'][start:end] = 300
            dt['Weight'][start:end] = 1
            dt['EventID'][start:end] = eventID[i]
            dt['ChannelID'][start:end] = channelID[i]
            start = start +1
        else:
            end = start + len(pet)
            dt['PETime'][start:end] = pet
            dt['Weight'][start:end] = pwe
            dt['EventID'][start:end] = eventID[i]
            dt['ChannelID'][start:end] = channelID[i]
            start = end
        print('\rThe analysis processing:|{}>{}|{:6.2f}%'.format(((20*i)//length)*'-', (19 - (20*i)//length)*' ', 100 * ((i+1) / length)), end=''if i != length-1 else '\n') # show process bar
    return dt[0:start]
def threshold_N(waveform, eventId, channelId, spe):
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint32), ('PETime', np.uint16), ('Weight', np.float)])
    moveDelta = np.where(np.max(spe))[0]
    waveform = waveform.astype(np.float)
    spe = spe.astype(np.float)
    length = waveform.shape[0]
    waveform /= np.sum(spe)
    dt = np.zeros(500* length, dtype=opdt)
    size = 0
    for i in range(length):
        recon, weight = threshold(waveform[i], moveDelta, 3)
        if len(recon) == 0:
            dt[size]['PETime'] = 300
            dt[size]['EventID'] = eventId[i]
            dt[size]['ChannelID'] = channelId[i]
            dt[size]['Weight'] = 1
            size += 1
        else:
            end = size + len(recon)
            if len(recon)>500:
                dt[size:(size+500)]['PETime'] = recon
                dt[size:(size+500)]['EventID'] = eventId[i]
                dt[size:(size+500)]['ChannelID'] = channelId[i]
                dt[size:(size+500)]['Weight'] = weight
                size += 500
            else:
                dt[size:end]['PETime'] = recon
                dt[size:end]['Weight'] = weight
                dt[size:end]['EventID'] = eventId[i]
                dt[size:end]['ChannelID'] = channelId[i]
                size = end
        print('\rThe analysis processing:|{}>{}|{:6.2f}%  channel{}size{}'.format(((20*i)//length)*'-', (19 - (20*i)//length)*' ', 100 * ((i+1) / length), channelId[i],size), end=''if i != length-1 else '\n') # show process bar
    print('size: {}'.format(size))
    return dt[0:size]
def threshold(waveform, moveDelta, sigma=3):
    waveform[waveform<0] = 0
    basethreshold = np.mean(waveform[0:100]) + sigma * np.std(waveform)
    recon = np.where(waveform>basethreshold)[0]
    weight = waveform[recon]
    return recon-moveDelta, weight
def lucyDDM_N(waveform, spe):
    signal = np.zeros(waveform.shape)
    length = waveform.shape[0]
    moveDelta = 10
    spe = np.append(np.zeros((spe.shape[0]- 2*moveDelta + 1,)), np.abs(spe))
    for i in range(length):
        signal[i, :] = lucyDDM(waveform[i, :], spe, 50)
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
    waveform = waveform.astype(np.float)
    spe = spe.astype(np.float)
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
def writefile(truth, subfile):
    with h5py.File(subfile, 'w') as opt:
        opt.create_dataset('Answer', data=truth, compression='gzip')
def writeSubfile(truth, eventId, channelId, sigma, moveDelta, subfile, mode='lack'):
    answerh5file = tables.open_file(subfile, mode='w', title="OneTonDetector")
    AnswerTable = answerh5file.create_table('/', 'Answer', AnswerData, 'Answer')
    answer = AnswerTable.row
    rangelist = np.arange(truth.shape[1])
    length = len(eventId)
    for i in range(length):
        eventIndex = eventId[i]
        channelIndex = channelId[i]
        basethreshold = np.mean(truth[i, 0:100]) + sigma * np.std(truth[i, 0:100])
        waitWriteTruth = rangelist[truth[i, :] > basethreshold]
        counter = 0
        for t in waitWriteTruth:# wait for the null output process
            if truth[i, t] > sigma * np.std(truth[i, (t-5):(t+5)]):
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
    opdt = np.dtype([('EventID', np.uint32), ('ChannelID', np.uint8), ('Waveform', np.uint16, truth.shape[1])])
    dt = np.zeros(len(eventId), dtype=opdt)
    dt['EventID'] = eventId
    dt['ChannelID'] = channelId
    dt['Waveform'] = truth
    with h5py.File(subfile,'w') as ipt:
        ipt.create_dataset('Recon', data=dt, compression='gzip')
def ReadWave(filename):
    h5file = h5py.File(filename, "r")
    waveform = h5file['Waveform']['Waveform']
    eventID = h5file['Waveform']['EventID']
    channelID = h5file['Waveform']['ChannelID']
    h5file.close()
    return (waveform, eventID, channelID)
if __name__ == "__main__":
    if len(sys.argv)<2:
        problemfile = 'dataset/juno/hdf5/12/wave.h5'
        subfile = 'output/juno/wave12/threshold/answer.h5'
        with h5py.File('output/juno/spe.h5', 'r') as ipt:
            spe = ipt['spe'][:]
        method = 'threshold'
        negativePulse=False
    else:
        problemfile = sys.argv[1]
        subfile = sys.argv[3]
        with h5py.File(sys.argv[2], 'r') as ipt:
            spe = ipt['spe'][:]
        method = sys.argv[4]
        negativePulse= sys.argv[5]
        if negativePulse=='True':
            negativePulse = True
        else:
            negativePulse = False
    '''
    spePart = spe[0:40].reshape((40,))
    if spePart[0]==0:
        spePart[0] = np.abs(spePart[1])
    '''
    (waveform, eventId, channelId) = ReadWave(problemfile)
    # need to be more careful
    numPMT = np.max(channelId)+1
    length = waveform.shape[1]
    '''
    waveformNobase = (np.sum(waveform[:, 0:150], axis=1)/150).reshape((waveform.shape[0], 1)).repeat(length, axis=1)- waveform
    if np.min(waveformNobase[0,:])<-10:
        waveformNobase = -waveformNobase
    '''
    print('Begin analyze {}, length:{}, pmt:{}'.format(problemfile, length, numPMT))
    # lucyTruth = lucyDDM_N(waveformNobase, spePart)
    reconTruth = recon_N(waveform, eventId, channelId,spe, method, negativePulse)
    if method == 'lucyddm':
        writeSubfile(reconTruth, eventId, channelId, 3, 9, subfile, 'full')
    else:
        writefile(reconTruth, subfile)
    print('End write {}'.format(subfile))
