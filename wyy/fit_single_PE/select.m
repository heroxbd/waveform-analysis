%生成一个PEtime表"pt2"，第EventID行第ChannelID行包含了一个对应波形PEtime的数组
pt2=cell(max(tTruth.EventID),30); %分配一个EventID总数个行，30列的cell数组内存空间，
for i=1:length(tTruth.EventID) %针对每个PEtime
    pt2{tTruth.EventID(i),tTruth.ChannelID(i)+1}(end+1)=tTruth.PETime(i); %在这个PE对应的PEtime数组末端添加这个PE的PEtime
end

%生成一个PENumbere表"PP"，第EventID行第ChannelID行包含了一个对应波形PE的个数
PP2 = int16(cellfun(@length,pt2));

%寻找单PE波形对应的EventID和ChannelID
[n,m] = find(PP2'==1); %在PE数表中寻找PE数为1的索引
mn = int32([m,n-1]); %这个索引分别是EventID和ChannelID，记录到一个2列矩阵中方便以后使用
