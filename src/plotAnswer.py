import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import sys

def viewTruth(waveform, truth, recon, spe, eventid, channelid, outdir, pdf, negativePulse=True):
    fig = plt.figure()
    fig.suptitle('eventid {}, channelid {}'.format(eventid, channelid))
    ax1 = fig.add_subplot(111)
    if negativePulse:
        waveformNobase = np.mean(waveform[-101:-1]) - waveform
    else:
        waveformNobase = waveform-np.mean(waveform[-101:-1])
    peCharge = np.sum(spe)
    ax1.plot(waveformNobase/peCharge, 'g')
    truth = np.array(truth)
    truth, count = np.unique(truth, return_counts=True)
    reconWaveform = np.zeros((waveform.shape[0]+len(spe),))
    for t,c in zip(truth, count):
        ax1.plot([t, t], [0, c], 'g')
    for r in recon:
        ax1.plot([r['PETime'], r['PETime']], [0, r['Weight']], 'r')
        reconWaveform[r['PETime']:(r['PETime']+len(spe))] += spe
    reconWaveform = reconWaveform[0:waveform.shape[0]]
    ax1.plot(reconWaveform/200, 'r')
    ax1.set_xlim([100, 500])
    # plt.savefig('{}/e{}c{}.png'.format(outdir, eventid, channelid))
    pdf.savefig(fig)
    plt.close()
if __name__ == "__main__":
    subfile = sys.argv[1]
    ansfile = sys.argv[2]
    spefile = sys.argv[3]
    outdir = sys.argv[4]
    eventid = int(sys.argv[5])
    channels = range(int(sys.argv[6]), int(sys.argv[7]))
    negativePulse = False
    pdf = PdfPages('{}/e{}c{}-{}.pdf'.format(outdir, eventid, sys.argv[6], sys.argv[7]))
    with h5py.File(ansfile,'r') as ref, h5py.File(subfile,'r') as subipt:
        with h5py.File(spefile, 'r') as speipt:
            spe = speipt['spe'][:]
        waveform = ref['Waveform'][:]
        groundtruth = ref['GroundTruth'][:]
        recontruth = subipt['Answer'][:]
        wave = [i for i in waveform if i['EventID']==eventid and i['ChannelID'] in channels]
        for i in range(len(wave)):
            truth = [t['PETime'] for t in groundtruth if t['EventID']==eventid and t['ChannelID']==channels[i]]
            recon = [t for t in recontruth if t['EventID']==eventid and t['ChannelID']==channels[i]]
            if len(truth)>0:
                print('channel {}'.format(channels[i]))
                viewTruth(wave[i]['Waveform'], truth, recon, spe, eventid, wave[i]['ChannelID'], outdir, pdf, negativePulse)
    pdf.close()
