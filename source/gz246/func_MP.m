function [d_ind,amp_mx] = func_MP(wave_input,atom_dic)
% this function implements the Matching Pursuit Algorithm on the particle
% detection problem in the THU-Jingping Project. 
% Input:wave_input : 1029*1 vector which is the orignal recorded waveform; 
% atom_dic: the time-shifted unit impulse reponse matrix; 
% Output: d_ind is a column vector stores the positions of the particles,
% e.g. d_ind = [208; 312] indicates at time 208 and 312 there is each
% particle. 
% amp_mx: a row vector that stores the amptitude of the particle. Usuaully
% not used in this version as we use equal weighting in all particles. But
% if we use the amptitude-related weighting scheme, this information is
% useful. 

wave_input = double(wave_input);
baseline =  round(nanmean(wave_input(1:100)));
wave_input = wave_input - baseline;

N = 1000;
res = wave_input;
amp_mx = nan(N,1);
d_ind = nan(N,1);

for i = 1:N
    s = res'*atom_dic;
    k1 = find(s==max(s));  % place;
    k2 = s(k1);  % amptitude
    if k2>7
        amp_mx(i) = k2;
        d_ind(i)= k1;
        res = res - amp_mx(i).*atom_dic(:,d_ind(i));
    else
        break;
    end
end

if i>1
    d_ind = d_ind(1:i-1);  % here it stops; throw out nan values
    amp_mx = amp_mx(1:i-1);
else
    d_ind = nan;
    amp_mx = nan;
end


end