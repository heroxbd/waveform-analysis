for k=1:10
filename = ['ztraining-',num2str(k-1),'.h5'];
InputTrainingFile; %输入训练集
select; %选择单PE波形
getallparameters; %获取所有单波形拟合参数
az = find(Para(:,3)==0); %找到cut条件之外的波形的编号
Para(az,:)=[]; %剔除cut条件之外波形对应的参数
mn(az,:)=[]; %剔除cut条件之外波形对应的EventID号与ChannelID号
PT{k,1}=pt2; %将这个文件的PEtime表记录在包含所有训练文件PEtime信息的大表中
PT{k,2}=PP2; %将这个文件的PE数记录在包含所有训练文件PEtime信息的大表中
Parameter{k,1}=Para; %将这个文件的单PE波形参数记录在包含所有训练文件单PE波形参数的大表中
Parameter{k,2}=mn; %将这个文件的单PE波形的EventID号与ChannelID号记录在包含所有训练文件单PE波形参数的大表中
indexing; %生成一个索引表，该索引表的第EventID行第ChannelID列是对应的WaveForm的索引号
Indexes{k} = int32(Indexes{k}); %将这个文件的索引表记录在包含所有训练文件索引表的大表中
end
save('Parameters.mat','Parameter','PT'); %保存PEtime大表和波形参数大表
save('Indexes.mat','Indexes'); %保存索引大表

AllPara = cell2mat(Parameter(:,1)); %所有10个训练文件的单PE波形参数合并至一个大矩阵
meanpara = mean(AllPara); %求平均波形参数