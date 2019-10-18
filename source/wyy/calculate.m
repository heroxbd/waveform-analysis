thresh=9; %设定阈值，超过该阈值的记作一个PE信号
len= size(TestWave.Waveform,2);
para=meanpara;
PEtime = cell(len,1); %预分配PE到达时间的内存，每个波形分配一个cell数组
WEIGHT = cell(len,1); %预分配权重的内存，每个波形分配一个cell数组

tic %开始计时
for i=1:len %对所有波形处理的循环
    thiswave = TestWave.Waveform(:,i)'; %把波形读取成1*1029的行向量到变量thiswave
    baseline = backgroundV(thiswave(1:200)); %选取前200ns波形算baseline
    thiswave = int16(baseline) - thiswave; %减去baseline
    if thiswave<thresh %如果没有任何波形过阈值，则选取波形最高点作为PEtime，同时权重为1
        tot=find(thiswave==max(thiswave)); %找到波形最高点时间为tot
        finalpetime = tot-7; %设定偏移量为7
        weigh = ones(1,length(finalpetime)); %权重为1
    else %如果有过阈信号
        begining = thiswave(1:10)<thresh; %检测前10ns波形有没有过阈（没有则全为1）
        if begining %全为1，即没有
            tot=0; %初始化tot
            finalpetime=[]; %初始化最终得到的petime
            weigh=[]; %初始化权重
        else
            tot=find(~begining,1); %如果前10ns中有过阈信号
            finalpetime = tot-6;% 找到它，设置偏移
            weigh = 1;
            thiswave(1:10)=0; %然后把前10ns信号设置为0。上述操作是防止有半个波形卡在前10ns导致程序bug。
        end
        while true %进入算法核心部分的循环：减去每个找到的波形
            wave = thiswave; %拷贝波形用于处理
            wave(wave<thresh)=0; %未过阈点认为是噪声，设为0
            tot = find(wave,1); %找到第一个过阈时间tot
            if isempty(tot) %如果再也找不到，宣告已全部找出所有信号
                break %因此结束循环
            end
            petime = tot-6; %如果找到了，那么减去偏移量得到这个信号的petime
            peaktime = findpeak(wave,1); %寻找第一个极大值
            if isempty(peaktime) %如果找不到极大值，则说明信号已经延申到波形外，认为是最后一个波形
                finalpetime = [finalpetime petime];
                weigh = [weigh 1];
                break
            end
            
            if isempty(finalpetime) || (petime-finalpetime(end))>4 || thiswave(peaktime)>12 %一些cut条件。有些
                weigh = [weigh,1];
                finalpetime=[finalpetime,petime];
            end
            if length(finalpetime)>500 %如果发现了超过有500个petime，应该是陷入了死循环，立即中止并报错。
                error('too many PEs found, there must be a bug.')
            end
            
            thiswave = thiswave - int16(modelfunc([petime+para(1),petime+para(2),para(3),para(4)],(1:1029))); %算法灵魂：在波形上减去已发现的信号对应的标准光电子波形，这样我们就可以将多个光电子信号叠加的波形中去除一个光电子波形。
            
        end %继续查找信号+减去已发现信号的循环
    end
    PEtime{i}=finalpetime;%结束一个波形的处理，写入光电子时间
    WEIGHT{i}=weigh;%结束一个波形的处理，写入权重
end
toc %结束计时
