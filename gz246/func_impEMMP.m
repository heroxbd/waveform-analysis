function [d_ind,amp_mx] = func_impEMMP(wave_input,atom_dic,theta)
% this function implements the improved EM-Matching Pursuit algorithm on
% the problem of signal detection for THU-Jingping project. 
% Input: wave_input is the single waveform for each recording, which is
% expected to be a 1*1029 vector;
% atom_dic is the time-shifted version of the unit impulse response, which
% is the output of the function template_training_final.m; it is a
% 1029*1029 matrix. 
% theta is a parameter that is used to determine whether a large peak is
% due to more than one particle.
% Output: d_ind is a column vector stores the positions of the particles,
% e.g. d_ind = [208; 312] indicates at time 208 and 312 there is each
% particle. 
% amp_mx: a row vector that stores the amptitude of the particle. Usuaully
% not used in this version as we use equal weighting in all particles. But
% if we use the amptitude-related weighting scheme, this information is
% useful. 

if nargin<3
    theta = 82;
end
large_thres = 45;

wave_input = double(wave_input);
[d_ind,amp_mx] = func_MP(wave_input,atom_dic);  % used for particle position initilization and number estimation; 

baseline =  round(nanmean(wave_input(1:100)));  
wave_input = wave_input - baseline;   % de-mean the raw signal; 


large_ind = amp_mx>large_thres;     % here we used a threshold to classify the peaks into two types: small amptitude and large amptitude
small_ind = length(find(amp_mx<=large_thres));  % large amptitude are more likely contain more than one particle. 

m = size(wave_input,1);
Energy = nansum(wave_input);
K= -min(Energy,0)./147;
K = round(K);     % use integration to calculate the number of particles. 

% three methods used to estimate the number of particles in total . 
cand1 = length(d_ind)+1;
cand2 = ceil(K);
cand3 = round(sum(amp_mx(large_ind))/45) + small_ind;   

% then a certain rule is designed to obtain the final estimate used in this
% program...depends on the number of K, then we use different ways to
% estimate the number to particles. 
if K<10
    S = ceil(mean([cand1, cand3]));
else
    S = median([cand1,cand2, cand3])+5;
end

if S<2
    return;
else
    d_ind_0 = d_ind;   % intitation for the d_ind0 and amp_mx0 for the EM procedue. 
    amp_mx_0 = amp_mx;
    n1 = length(d_ind);

    d_ind = nan(S,1);   % set the output as NaNs here. 
    amp_mx = nan(S,1);
    res = wave_input;
    D_previous = sum(res.^2);   % record the residues energy, stored in D_previous; 
    cd = zeros(m,S); 
    
    % if number of previous detected particles is less than number of particles estimated here, 
    % use the previous results as initialition; otherwise skip the
    % initialization; 
    if n1<S        
        cd(:,1:n1) = repmat(amp_mx_0',m,1).*atom_dic(:,d_ind_0);
        d_ind(1:n1) = d_ind_0;
        amp_mx(1:n1) = amp_mx_0;
    end
    
    
    tt = 1;   % record the current loop number...
    
    %% here we use a special-designed method to speed-up the calculation of matrix mutliplication;
    % don't need to dig into them; 
    % they are just trying to calculate the s = atom_dic'*y_temp_i...
    D_current_store = nan(S,1);
    A_row = [atom_dic(1,1),zeros(1,m-1)];
    
    c=[A_row'; 0; flip(atom_dic(2:end,1))];    
    fc = fft(c);

    MaxIt =50;
    while tt<MaxIt
        for i = 1:S
            y_temp_i = res - sum(cd,2) + cd(:,i);      % this is the residules for the i-th component ;
            s = real(toeplitzmult_gz(fc,y_temp_i));
            s = s(1:1029);
            % all above lines are just trying to calculate s = atom_dic'*y_temp_i...
            % using toeplitzmult and FFt to make it faster...
            
            k1 = find(s==max(s));  % place;
            if length(k1)>1
                k1 = k1(1);
            end
            % we are using the non-positive restriction to explore the
            % local solution which would lower the error in terms of
            % maximum recontructed signal level...
            if (k1>1)&&(k1+1<1029)
                temp2 = s((k1-1):(k1+1));
                temp3 = floor(temp2./theta)+1;
                temp2 = temp2./temp3;

                p = atom_dic(:,(k1-1):(k1+1));

                res_temp1 = y_temp_i - p(:,1)*temp2(1);            
                res_temp2 = y_temp_i - p(:,2)*temp2(2);            
                res_temp3 = y_temp_i - p(:,3)*temp2(3);     
                d1 = max(0,max(res_temp1-4.4));
                d2 = max(0,max(res_temp2-4.4));
                d3 = max(0,max(res_temp3-4.4));
                DD = [d1,d2,d3];
                if (s(k1)<42)|((max(DD)==0)|(d2 ==min(DD)))
                    k2 = k1;
                else
                    if d1==min(DD)
                        k2 = k1-1;
                    else
                        k2 = k1+1;
                    end
                end
            else
                k2 = k1;
            end
            r1 = floor(s(k2)/theta);
            if r1==0
                amp_mx(i) = s(k2);
            else
                amp_mx(i) = s(k2)./(r1+1);
            end
            p = atom_dic(:,k2);
            cd(:,i) = p*amp_mx(i);
            d_ind(i) = k2;
            D_current_store(i,1) = sum((res - sum(cd,2)).^2);
        end
        D_current = mean(D_current_store);
        if D_current<D_previous
            D_previous = D_current;
            tt = tt+1;
        else
            break;
        end
    end

    if S > 10
        thres = 8.5;
    else
        thres = 6.5;
    end
    
    
SS = find(amp_mx>thres);
d_ind = d_ind(SS);
amp_mx = amp_mx(SS);

end

end



function y=toeplitzmult_gz(fc,x)

n= 1029;

p=ifft(fc.*fft([x; zeros(n,1)]));

y=p(1:n);

end
