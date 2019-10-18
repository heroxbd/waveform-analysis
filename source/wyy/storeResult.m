EventID=[];
ChannelID=[];
PETime=[];
Weight=[];

for i = 1:len
    eventid=TestWave.EventID(i);
    channelid=TestWave.ChannelID(i);
    tmplen=size(PEtime{i},2);
    PETime=[PETime,PEtime{i}];
    Weight=[Weight,WEIGHT{i}];
    EventID=[EventID,eventid+int64(zeros(1,tmplen))];
    ChannelID=[ChannelID,channelid+int16(zeros(1,tmplen))];
end
answer1=struct('EventID',EventID','ChannelID',ChannelID','PETime',PETime','Weight',Weight');
save(['DeltaMethod',filename,'.mat'],'answer1');