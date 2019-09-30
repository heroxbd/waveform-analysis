function [peaktime] = findpeak(wave,n)
%findpeak 寻找向量的极大值点
%   寻找向量的极大值点并返回最大值点的位置，如有平坦的极大值区则寻找下降沿
%   示例：
%   findpeak([1,2,4,2,1])
% 
%    ans =
% 
%         3
%   findpeak([1,2,3,4,4,4,2,3,3,1])
% 
%    ans =
% 
%         6     9
if nargin==1
    peak = int8(sign(diff(wave)));
    I=1:length(wave);
    I(peak==0)=[];
    peak(peak==0)=[];
    peak = diff(peak);
    b = find(peak==-2)+1;
    peaktime=I(b);
elseif nargin==2
    peak = int8(sign(diff(wave)));
    I=1:length(wave);
    I(peak==0)=[];
    peak(peak==0)=[];
    peak = diff(peak);
    b = find(peak==-2,n)+1;
    peaktime=I(b);
end
end

