function [backg] = backgroundV(wave)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
wave(wave<970)=[];
backg=mean(wave);
end

