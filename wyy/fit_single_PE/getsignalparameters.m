function [parameters] = getsignalparameters(eventid,channelid,petime)
%getsignalparameters 拟合单PE波形的波形参数
%   根据给出的标准波形'modelfunc'拟合单PE波形的参数
Wave = evalin('base','tWave'); %从工作区读取文件
fitfunc = evalin('base','modelfunc'); %从工作区读取拟合函数
thiswaveevntid = find(Wave.EventID==eventid); 
thiswave = (Wave.Waveform(:,thiswaveevntid(Wave.ChannelID(thiswaveevntid)==channelid)))'; %寻找到该波形
if min(thiswave)<963 %cut条件：过小波形不予以考虑
    time = double(petime);
    fitregion = (time-10):(time+30); %选取拟合区域
    nothing = double(thiswave(1:90)); 
    backg = mean(nothing); %计算baseline
    fitwave = double(thiswave(fitregion)); %选取待拟合波形
    % parameters = nlinfit(fitregion,backg-fitwave,fitfunc,[time+5,time+9,time+9,39.5761,4.1538]); %拟合
    parameters = nlinfit(fitregion,backg-fitwave,fitfunc,[time+5,time+9,39.5761,4.1538]);
    % parameters(1:3) = parameters(1:3) - time; %para1
    parameters(1:2) = parameters(1:2) - time;
else
    parameters = [0 0 0 0]; %不满足cut条件则记为全0
end

