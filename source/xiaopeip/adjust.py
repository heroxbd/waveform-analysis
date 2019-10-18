import h5py
import numpy as np
import tables
#使用tables输出结果
class AnswerData(tables.IsDescription):
    EventID    = tables.Int64Col(pos=0)
    ChannelID  = tables.Int16Col(pos=1)
    PETime     = tables.Int16Col(pos=2)
    Weight     = tables.Float32Col(pos=3)

h5file = tables.open_file("finalanswer.h5", mode="w", title="OneTonDetector")
# Create tables
AnswerTable = h5file.create_table("/", "Answer", AnswerData, "Answer")
answer = AnswerTable.row

#读入文件
with h5py.File("result/Total.h5") as ipt:
    yuanru=ipt["Answer"][()]

#按照event和channel分类，获取分界的节点
eve_p=yuanru[0]["EventID"]
cha_p=yuanru[0]["ChannelID"]
jiedian=[0]
length=len(yuanru)
for n in range(length):
    if yuanru[n]["EventID"]!=eve_p or yuanru[n]["ChannelID"]!=cha_p:
        jiedian.append(n)
        eve_p,cha_p=yuanru[n]["EventID"],yuanru[n]["ChannelID"]
    if n%100000==0:
        print(n)
jiedian.append(length)

#a1,a2,a3是叠加时的系数
a1,a2,a3=0.9,1.7,0.9
for i in range(len(jiedian)-1):
    if i%10000==0:
        print(i)
    copy=yuanru[jiedian[i]:jiedian[i+1]].copy()

#用copy减去舍入后的结果得到零头进行后面的计算
    sheru=np.around(copy["Weight"])
    copy["Weight"]=copy["Weight"]-sheru
#a的作用是获取数组的长度
    a=np.unique(np.array([[x-1 for x in copy["PETime"]],[x for x in copy["PETime"]],[x+1 for x in copy["PETime"]]]))
    leftright=np.zeros(len(a)+1,dtype=[("PETime",np.int),("Weight",np.float)])
    k=0
#将左右两点的零头和本身的零头乘以不同系数相加
    for j in range(len(copy)):
        if k>1 and leftright[k-2]["PETime"]==copy[j]["PETime"]-1:
            leftright[k-2]["Weight"]+=a1*copy[j]["Weight"]
        elif k>0 and leftright[k-1]["PETime"]==copy[j]["PETime"]-1:
            leftright[k-1]["Weight"]+=a1*copy[j]["Weight"]
        else :
            leftright[k]["PETime"]=copy[j]["PETime"]-1
            leftright[k]["Weight"]+=a1*copy[j]["Weight"]
            k+=1
        if k>0 and leftright[k-1]["PETime"]==copy[j]["PETime"]:
            leftright[k-1]["Weight"]+=a2*copy[j]["Weight"]
        else:
            leftright[k]["PETime"]=copy[j]["PETime"]
            leftright[k]["Weight"]+=a2*copy[j]["Weight"]
            k+=1
        leftright[k]["PETime"]=copy[j]["PETime"]+1
        leftright[k]["Weight"]+=a3*copy[j]["Weight"]
        k+=1

#若结果大于0.5且大于相邻两点的结果则可能需要补加权重,将这些时刻存入plustime
    plustime=[]
    while True:
        newp=np.argmax(leftright["Weight"])
        if leftright[newp]["Weight"]>0.5:
            plustime.append(leftright[newp]["PETime"])
            leftright[newp]["Weight"]=0
            if leftright[newp+1]["PETime"]==leftright[newp]["PETime"]+1:
                leftright[newp+1]["Weight"]=0
            if leftright[newp-1]["PETime"]==leftright[newp]["PETime"]-1:
                leftright[newp-1]["Weight"]=0
        else:
            break

#先将之前舍入后的结果输出
    count=0
    for i1 in range(len(sheru)):
        if sheru[i1]>0:
            count+=1
            answer['EventID'] =  copy[0]["EventID"]
            answer['ChannelID'] = copy[0]["ChannelID"]
            answer['PETime'] = copy[i1]["PETime"]
            answer['Weight'] = sheru[i1]
            answer.append()
#如果plustime中的点此前无输出（即之前舍入时被舍掉了），则补加权重输出
    for i2 in plustime:
        posi=np.where(copy["PETime"]==i2)[0]
        if len(posi)==0 or sheru[posi[0]]==0:
            
            answer['EventID'] =  copy[0]["EventID"]
            answer['ChannelID'] = copy[0]["ChannelID"]
            answer['PETime'] = i2
            answer['Weight'] = 1.
            answer.append()
#如果修改后事例为空，则保留原有的输出
    if len(plustime)+count==0:
        for i3 in range(jiedian[i],jiedian[i+1]):
            answer['EventID'] =  copy[0]["EventID"]
            answer['ChannelID'] = copy[0]["ChannelID"]
            answer['PETime'] = yuanru[i3]["PETime"]
            answer['Weight'] = 4*(yuanru[i3]["Weight"])**2
            answer.append()

AnswerTable.flush()
h5file.close()
