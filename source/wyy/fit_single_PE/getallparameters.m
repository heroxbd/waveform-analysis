%拟合波形参数，拟合函数是两个分段函数，RC充电和RC放电
%第一个波形参数Para(1)：从PEtime开始起过多长时间电压开始下降
%第二个波形参数Para(2)：从PEtime开始起过多长时间电压达到最小值
%第三个波形参数Para(3)：振幅（与峰值成正比但差一个系数）
%第四个波形参数Para(4)：RC系数
modelfunc = @(A,t)A(3)*((1-exp(-(t-A(1))./A(4))).*(t<=A(2)).*(t>=A(1))+(exp(-(A(1)-A(2))./A(4))-1).*exp(-(t-A(1))./A(4)).*(t>A(2))); %拟合函数
% Para = zeros(length(mn),5); %预分配空间
Para = zeros(length(mn),4);
for i=1:length(mn) %针对每一个单PE波形
petime = pt2{mn(i,1),mn(i,2)+1}; %从PEtime表中读出真实PEtime
    if (100<petime && petime<999) %cut条件：如果这个波形是完整的，即不在头和尾
        Para(i,:)=getsignalparameters(mn(i,1),mn(i,2),petime); %拟合波形参数
    end
end