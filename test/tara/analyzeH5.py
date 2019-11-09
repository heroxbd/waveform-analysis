import tables
import numpy as np
import math
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import h5py
#define the format of result file
class AnswerData(tables.IsDescription):
    EventID=tables.Int64Col(pos=0)
    ChannelID=tables.Int16Col(pos=1)
    PETime=tables.Int16Col(pos=2)
    Weight=tables.Float32Col(pos=3)
#from the mat file to the h5 file
def mat2h5(fin,fout):
    from scipy.io import loadmat
    answer=loadmat(fin)
    answer=answer['answer1']
    EventID=answer['EventID'][0][0]
    ChannelID=answer['ChannelID'][0][0]
    PETime=answer['PETime'][0][0]
    Weight=answer['Weight'][0][0]
    opd=[('EventID', '<i8'), ('ChannelID', '<i2'),('PETime', '<i2'), ('Weight', 'f4')]
    with h5py.File(fout,"w") as opt:
        dt=np.zeros(len(EventID),dtype=opd)
        dt['EventID']=EventID.reshape(len(EventID))
        dt['ChannelID']=ChannelID.reshape(len(EventID))
        dt['PETime']=PETime.reshape(len(EventID))
        dt['Weight']=Weight.reshape(len(EventID))
        opt.create_dataset('Answer',data=dt,compression='gzip')

#Read hdf5 file
def ReadWave(filename):
    h5file=tables.open_file(filename,"r")
    waveTable=h5file.root.Waveform
    entry=0
    wave=waveTable[:]['Waveform']
    eventId=waveTable[:]['EventID']
    channelId=waveTable[:]['ChannelID']
    h5file.close()
    return (wave,eventId,channelId)
def ReadTruth(filename):
    h5file=tables.open_file(filename,"r")
    waveTable=h5file.root.GroundTruth
    entry=0
    PETime=waveTable[:]['PETime']
    truthchannelId=waveTable[:]['ChannelID']
    h5file.close()
    return (PETime,truthchannelId)
#analyze wave and write the result h5file
def analyzeWave(waveform,eventId,channelId,delta,outname):
    numPMT=30
    length=1029
    #eventRange=range(0,max(eventId))
    #max(eventId))
    truth=[]
    weight=[]
    temptruth=[]
    tempweight=[]
    H=len(eventId)
    #refer to the platform,build up the file
    answerh5file = tables.open_file(outname, mode="w", title="OneTonDetector")
    AnswerTable = answerh5file.create_table("/", "Answer", AnswerData, "Answer")
    answer = AnswerTable.row
    
    for index in range(H):
        #print(eventIndex)
        #for pmtIndex in range(0,numPMT):
        eventIndex=eventId[index]
        pmtIndex=channelId[index]
        truth,weight=analyzePMT(waveform[index,:],length,5)
        if not truth:
            truth,weight=analyzePMT(waveform[index,:],length,3)
            
        if not truth or not weight:
            truth=temptruth
            weight=tempweight
            print("warning none",eventIndex,pmtIndex)
        else:
            temptruth=truth
            tempweight=weight
        for t,w in zip(truth,weight):
            answer['EventID'] = eventIndex
            answer['ChannelID'] = pmtIndex
            answer['PETime'] = t+delta
            if w<=0:
                print("warning negative",eventIndex,pmtIndex,w)
                w=1
            answer['Weight'] = w            
            answer.append()
    AnswerTable.flush()
    answerh5file.close()
#analyze wave and write the result h5file
def analyzefftWave(waveform,eventId,channelId,tau,delta,outname,SPE):
    numPMT=30
    length=1024
    #eventRange=range(0,max(eventId))
    #max(eventId))
    truth=[]
    weight=[]
    temptruth=[]
    tempweight=[]
    H=len(eventId)
    #refer to the platform,build up the file
    answerh5file = tables.open_file(outname, mode="w", title="OneTonDetector")
    AnswerTable = answerh5file.create_table("/", "Answer", AnswerData, "Answer")
    answer = AnswerTable.row
    
    for index in range(H):
        #for pmtIndex in range(0,numPMT):
        
        eventIndex=eventId[index]
        pmtIndex=channelId[index]
        #if(eventIndex==63):
        #    break
        truth,weight=analyzefftPMT(waveform[index,:],length,5,tau,SPE)
        if not truth:
            print(truth,temptruth)
            truth,weight=analyzefftPMT(waveform[index,:],length,3,tau,SPE)
        if not truth:
            #print(truth,temptruth)
            truth=temptruth
            weight=tempweight
            #print("none",eventIndex,pmtIndex)
        else:
            temptruth=truth
            tempweight=weight
        if not truth:
            print("error",eventIndex,pmtIndex)
            break
        for t,w in zip(truth,weight):
            answer['EventID'] = eventIndex
            answer['ChannelID'] = pmtIndex
            answer['PETime'] = t+delta
            if w<=0:
                print("negative",eventIndex,w)
                w=1
            answer['Weight'] = w
            if eventIndex==7455 &pmtIndex==0:
                print(eventIndex,pmtIndex,truth,weight)
            answer.append()
    AnswerTable.flush()
    answerh5file.close()
#analyze each PMTwave
def analyzePMT(waveform,length,multiSigma):
    baseline=np.mean(waveform[0:10])
    sigma=np.std(waveform[0:10],ddof=1)
    threshold=baseline-multiSigma*sigma

    statemachine=0
    truth=[]
    weight=[]
    if multiSigma>3:
        for i in range(10,length-10):
            if statemachine==0:
                if waveform[i]<threshold and (waveform[i+1]<waveform[i]) and waveform[i+2]<waveform[i+1] and waveform[i+3]<waveform[i+2]:
                    statemachine=1
                    upslope=(waveform[i]-waveform[i+3])/3
                    truth.append(i-math.ceil((baseline-waveform[i])/upslope))
                
            elif statemachine==1:
                if (waveform[i+1]>waveform[i])and waveform[i+2]>waveform[i+1] and waveform[i+3]>waveform[i+2]:
                    statemachine=2
                    tempweight=math.ceil((baseline-waveform[i])/10)
                    if tempweight<=0:
                        tempweight=1
                    weight.append(tempweight)
            elif statemachine==2:
                if (waveform[i]>threshold )and waveform[i+1]>threshold and waveform[i+2]>threshold:
                    statemachine=0
                elif (waveform[i+1]<waveform[i])and waveform[i+2]<waveform[i+1] and waveform[i+3]<waveform[i+2] and waveform[i+4]<waveform[i+3]:
                    statemachine=1
                    downslope=(waveform[i-2]-waveform[i])/2
                    upslope=(waveform[i]-waveform[i+3])/3
                    if downslope<0 :
                        truth.append(i-math.ceil((baseline-waveform[i])/(upslope-2*downslope)))
                    else:
                        truth.append(i-math.ceil((baseline-waveform[i])/(upslope)))
    else:
        for i in range(10,length-10):
            if statemachine==0:
                if waveform[i]<threshold and (waveform[i+1]<waveform[i]) and waveform[i+2]<waveform[i] :
                    statemachine=1
                    upslope=(waveform[i]-waveform[i+2])/2
                    truth.append(i-math.ceil((baseline-waveform[i])/upslope))
                
            elif statemachine==1:
                if (waveform[i+1]>waveform[i])and waveform[i+2]>waveform[i+1]:
                    statemachine=2
                    tempweight=math.ceil((baseline-waveform[i])/10)
                    if tempweight<=0:
                        tempweight=1
                    weight.append(tempweight)
            elif statemachine==2:
                if (waveform[i]>threshold )and waveform[i+1]>threshold and waveform[i+2]>threshold:
                    statemachine=0
                elif (waveform[i+1]<waveform[i])and waveform[i+2]<waveform[i+1] and waveform[i+3]<waveform[i+2] :
                    statemachine=1
                    downslope=(waveform[i-2]-waveform[i])/2
                    upslope=(waveform[i]-waveform[i+3])/3
                    if downslope<0 :
                        truth.append(i-math.ceil((baseline-waveform[i])/(upslope-2*downslope)))
                    else:
                        truth.append(i-math.ceil((baseline-waveform[i])/(upslope)))
    return (truth,weight)
#def the generator of exp
def exp_gen(length,amplitude,tau):
    return np.multiply(np.exp(np.true_divide(range(0,length),0-tau)),amplitude)
#analyze each PMTwave use the small template and use fft
def analyzefftPMT(waveform,length,multiSigma,tau,sample):
    #use fft to preanalyze the wave
    baseline=np.mean(waveform[0:10])
    data=baseline-waveform[0:length]
    dataf=fft(data,length)
    #sample=exp_gen(length,1,tau)
    #sample=np.load("SPE.npy")
    
    samplef=fft(sample,length)
    plt.plot(sample)
    plt.show()
    signalf=np.true_divide(dataf,samplef)
    signal=np.real(ifft(signalf,length))
    plt.plot(signal)
    plt.show()
    print(sample)
    #tell from signal
    sigma=np.std(signal[0:10],ddof=1)
    threshold=multiSigma*sigma

    statemachine=0
    truth=[]
    weight=[]
    weighttemp=0
    for i in range(10,length-10):
        if statemachine==0:
            if signal[i]>threshold and (signal[i+1]>signal[i]) and signal[i+2]>signal[i] :
                statemachine=1
                truth.append(i)
                weighttemp+=signal[i]
        elif statemachine==1:
            
            if (signal[i+1]<signal[i]) :
                statemachine=2
                weighttemp+=signal[i]  
        elif statemachine==2:
            if (signal[i]<threshold )and signal[i+1]<threshold :
                statemachine=0
                weight.append(weighttemp/10)
                weighttemp=0
            elif (signal[i+1]>signal[i])and signal[i+2]>signal[i] :
                statemachine=1
                weight.append(weighttemp/10)
                weighttemp=signal[i]
                truth.append(i)
            if signal[i]>threshold:
                weighttemp+=signal[i]
    return (truth,weight)
#general function read and generate the result
def output(filename,outname):
    (waveform,eventId,channelId)=ReadWave(filename)
    analyzeWave(waveform,eventId,channelId,0,outname)

#
def fftoutput(filename,tau,outname,SPE):
    (waveform,eventId,channelId)=ReadWave(filename)
    delta=0
    analyzefftWave(waveform,eventId,channelId,tau,delta,outname,SPE)

#test the ngroups of the answer
def testAnswer(filename):
    import pandas as pd
    import h5py
    f_sub=h5py.File(filename)
    df_sub=pd.DataFrame.from_records(f_sub['Answer'][()])
    subg=df_sub.groupby(['EventID','ChannelID'],as_index=True)
    print(subg.ngroups)
    f_sub.close()
#test the wave distribution w-distance with the truth
def wave2sub(waveform,eventId,channelId,fftif,multisig,delta,outputname):
    numPMT=30
    length=1029
    truth=[]
    weight=[]
    temptruth=[]
    tempweight=[]
    H=len(eventId)
    #refer to the platform,build up the file
    answerh5file = tables.open_file(outputname, mode="w", title="OneTonDetector")
    AnswerTable = answerh5file.create_table("/", "Answer", AnswerData, "Answer")
    answer = AnswerTable.row
    if fftif==0:
        for index in range(H):
        #print(eventIndex)
        #for pmtIndex in range(0,numPMT):
            eventIndex=eventId[index]
            pmtIndex=channelId[index]
            baseline=np.mean(waveform[index,1:10])
            truth=baseline-waveform[index,0:length]
            sigma=np.std(truth[0:10])
            for i in range(length):
                if truth[i]>multisig*sigma:
                    answer['EventID'] = eventIndex
                    answer['ChannelID'] = pmtIndex
                    answer['PETime'] = i+delta
                    answer['Weight'] = truth[i]            
                    answer.append()
    else:
        print("fft")
        #spe=np.load("SPE.npy")
        from scipy.io import loadmat
        import os
        spe=loadmat("SPEAd.mat")['SPE'].reshape(1029)
        for index in range(H):
        #print(eventIndex)
        #for pmtIndex in range(0,numPMT):
            eventIndex=eventId[index]
            pmtIndex=channelId[index]
            truth=getWavefft(waveform[index,:],1024,2,-spe,range(400,1024*2-400))
            sigma=np.std(truth[0:50])
            top=np.max(truth)
            while top<multisig*sigma*3 and multisig>7:
                multisig=multisig-1
            thre=0.13
            #if multisig*sigma>0.18:
            #    thre=multisig*sigma
            #print(thre)
            counter=0
            for i in range(200,900):
                if truth[i]>thre and truth[i]>np.std(truth[(i-5):(i+5)]):
                    answer['EventID'] = eventIndex
                    answer['ChannelID'] = pmtIndex
                    answer['PETime'] = i+delta
                    if(truth[i]<0.3 ):
                        answer['Weight'] = 0.3
                    else:
                        answer['Weight'] = truth[i]            
                    answer.append()
                    counter+=1
            if counter==0:
                answer['EventID'] = eventIndex
                answer['ChannelID'] = pmtIndex
                answer['PETime'] = 300
                answer['Weight'] = 1            
                answer.append()
            '''
            for i in range(200,900):
                if truth[i]>thre and truth[i]>truth[i-1] and truth[i]>truth[i+1]:
                    intThre=truth[i]/2
            '''
    AnswerTable.flush()
    answerh5file.close()
#analyze each PMTwave use the small template and use fft and return the filter wave
def getWavefft(waveform,length,tau,spe,Range):
    #use fft to preanalyze the wave,spe is response,range is the cut
    baseline=np.mean(waveform[0:150])
    data=baseline-(waveform[0:length])
    dataf=fft(data,2*length)
    #sample=exp_gen(length,1,tau)
    sample=spe
    dataftemp=dataf.copy()
    dataftemp[0:length]=dataf[length:2*length]
    dataftemp[length:2*length]=dataf[0:length]
    x=np.array([i for i in range(2*length)])
    dataftemp=np.multiply(dataftemp,np.exp(-(np.multiply(x-length,x-length))/2/(200**2)))
    dataf[0:length]=dataftemp[length:2*length]
    dataf[length:2*length]=dataftemp[0:length]

    samplef=fft(sample,2*length)
    signalf=np.true_divide(dataf,samplef)
    
    # signalf[range]=0
    signal=np.real(ifft(signalf,2*length))
    #plt.plot(signal)
    return signal
#get the SPE response use foolish method
def getSPEResFool(data,height,length,multiSigma):
    baseline=np.mean(data[0:10])
    waveform=baseline-data[0:length]
    sigma=np.std(waveform[0:10],ddof=1)
    threshold=multiSigma*sigma
    truth=np.zeros(length)
    statemachine=0
    startTime=0
    endTime=0
    upslope=0
    downslope=0
    for i in range(10,length-10):
        if statemachine==0:
            if waveform[i]>threshold and (waveform[i+1]>waveform[i]) and waveform[i+2]>waveform[i+1] and waveform[i+3]>waveform[i+2]:
                statemachine=1
                upslope=(waveform[i+3]-waveform[i])/3
                startTime=(i-math.ceil((waveform[i])/upslope))        
        elif statemachine==1:
            if (waveform[i+1]<waveform[i])and waveform[i+2]<waveform[i+1] and waveform[i+3]<waveform[i+2]:
                statemachine=2
        elif statemachine==2:
            if (waveform[i]<threshold )and waveform[i+1]<threshold and waveform[i+2]<threshold:
                statemachine=0
                endTime=i
                break
            elif (waveform[i+1]>waveform[i])and waveform[i+2]>waveform[i+1] and waveform[i+3]>waveform[i+2]:
                statemachine=1
                endTime=i
                downslope=(waveform[i]-waveform[i-2])/2
                break
    truth[0:endTime-startTime]=waveform[startTime:endTime]
    if downslope!=0:
        for i in range(endTime-startTime+1,length):
            if truth[i-1]>downslope+threshold:
                truth[i]=truth[i-1]-downslope
            else:
                break
    truth=truth/height
    print(startTime,endTime)
    np.save("SPE",truth)
    return truth
#return the SPE response by different methods
def getSPERes(filename):
    (waveform,eventId,channelId)=ReadWave(filename)
    truth=getSPEResFool(waveform[0,:],8,1024,5)
    import matplotlib.pyplot as plt
    plt.plot(truth)
    plt.show()
