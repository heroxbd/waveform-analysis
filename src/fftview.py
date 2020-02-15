import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
from scipy.fftpack import fft, ifft

def viewTruth(waveform, truth, spe, eventid, channelid, outdir, pdf, negativePulse=True):
    length =waveform.shape[0]
    spefft = fft(spe, 2*length)
    wavef = fft(waveform, 2*length)
    wavef[(length-int(length*0.7)):(length+int(length*0.7))] = 0
    signalf = np.true_divide(wavef, spefft)
    reconWaveform = np.real(ifft(signalf, 2*length))

    fig = plt.figure()
    fig.suptitle('eventid {}, channelid {}'.format(eventid, channelid))
    ax1 = fig.add_subplot(111)
    if negativePulse:
        waveformNobase = np.mean(waveform[0:100]) - waveform
    else:
        waveformNobase = waveform-np.mean(waveform[0:100])
    ax1.plot(waveformNobase/np.sum(spe), 'g')
    truth = np.array(truth)
    truth, count = np.unique(truth, return_counts=True)
    for t,c in zip(truth, count):
        ax1.plot([t, t], [0, c], 'g')
    ax1.plot(reconWaveform, 'r')
    ax1.set_xlim([np.min(truth)-50, np.max(truth)+50])
    # plt.savefig('{}/e{}c{}.png'.format(outdir, eventid, channelid))
    pdf.savefig(fig)
    plt.close()

if __name__== "__main__":
    spefile = sys.argv[1]
    ansfile = sys.argv[2]
    outdir = sys.argv[3]
    eventid = int(sys.argv[4])
    channelidb = sys.argv[5]
    channelide = sys.argv[6]
    channels = range(int(channelidb), int(channelide))
    negativePulse=False
    print('{}Origine{}c{}-{}.pdf'.format(outdir, eventid, channelidb, channelide))
    pdf = PdfPages('{}Origine{}c{}-{}.pdf'.format(outdir, eventid, channelidb, channelide))
    with h5py.File(spefile) as speipt, h5py.File(ansfile) as ansipt:
        waveform = ansipt['Waveform'][:]
        groundtruth = ansipt['GroundTruth'][:]
        spe = speipt['spe'][:]
        
        wave = [i for i in waveform if i['EventID']==eventid and i['ChannelID'] in channels]
        print(len(wave))
        for i in range(len(wave)):
            truth = [t['PETime'] for t in groundtruth if t['EventID']==eventid and t['ChannelID']==channels[i]]
            if len(truth)>0:
                print('channel {}'.format(channels[i]))
                viewTruth(wave[i]['Waveform'], truth, spe, eventid, wave[i]['ChannelID'], outdir, pdf, negativePulse)
    pdf.close()
        