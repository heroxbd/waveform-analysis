%生成一个索引表表"indexee"，第EventID行第ChannelID行包含了一个对应波形的索引
indexee = zeros(max(tTruth.EventID),30);
for i=1:length(tWave.EventID)
indexee(tWave.EventID(i),tWave.ChannelID(i)+1)=i;
end