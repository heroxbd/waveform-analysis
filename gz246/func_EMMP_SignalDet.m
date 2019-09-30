function [awtable] = func_EMMP_SignalDet(target_file,output_file,trainif,training_file)
% this file will generate the answer table from the EMMP (expectation
% maximization matching pursuit) algorithm;
% Input variables: 
% target_file: the filename(with path) from which signals are to be detected;
% training_file: a training file from which template of unit pulse response
% can be estimated;
% output_file: the name of the file to which answer table containg the position and weight for the
% detected signals can be writen.  
% Output variable:
% awtable: A table structure that contains the positions and weights for
% the signals detected. 
% Example: [awtable] = my_answer_v4();
%% Example2(run the following lines):
% target_file = 'D:\proj_signalDetection\zincm-problem.h5';
% training_file = 'D:\proj_signalDetection\ftraining-6.h5';
% output_file - 'myAnswer.txt';
% [awtable] = my_answer_v4(target_file,training_file,output_file);

%% Explaination: 
% the content in output (`awtable') will be written on the
% output file 'myAnswer.txt'. This makes the python processing easier. 
%% decide the targeting file that is to be processed...
if nargin <1
    target_file = 'D:\proj_signalDetection\p1data.h5';   % this is faster for testing purpose. 
%% or you can use the following file to make the formal test. replace it with the previous line. 
%   target_file = 'D:\proj_signalDetection\zincm-problem.h5';  
end

if nargin <2
    training_file = 'D:\proj_signalDetection\ftraining-6.h5';
end

if nargin <3
    output_file = 'D:\proj_signalDetection\myAnswer.txt';
end

if trainif==1
    [atom_dictionary] = template_training_final(50000,training_file);
    return
end

if trainif==0   % check if the atom_dictionary exists/defined;
    try 
        load('atom_dictionary.mat');  % try to load the atom_dictionary from local path;
    catch
        % if atom_dictionary has not been calculated, then calculate it
        % from the training dataset. 
        [atom_dictionary] = template_training_final(50000,training_file);
    end
end
%% load the data from the target file; 
data_waveform = h5read(target_file,['/','Waveform']);
M = length(data_waveform.EventID);

data_Answer.EventID = int64(nan(M.*30,1));
data_Answer.ChannelID = int16(nan(M.*30,1));
data_Answer.PETime = int16(nan(M.*30,1));
data_Answer.Weight = single(nan(M.*30,1));

k = 1;
tic
for j = 1:M

    x = data_waveform.Waveform(:,j);
    
    % this is the main impEMMP algorithm that we are using; 
    [d_ind,amp_mx] = func_impEMMP(x,atom_dictionary,82);
    
    N_answer = length(d_ind);
    flag_status = false;
    
    % only if the output is empty, we use the MP algorithm. 
    if N_answer==0
        [d_ind,amp_mx] = func_MP(x,atom_dictionary);
        flag_status = true;   % this flag tells if the addtional reweight step is needed later. 
    end
   

    eventID_wvf = data_waveform.EventID(j);
    ChanID_wvf = data_waveform.ChannelID(j);

    f2 = d_ind;  % answer obtained;
    %% calculate the weight
    w2 = ones(N_answer,1);   % by default (EMMP case) we just use the equal weighting. 
    
    if flag_status   % the MP algorithm will need to have a reweight step as it does not restrict the maximum energy for any single event. 
        w2 = 1/N_answer.*ones(N_answer,1);
        large_ind = find(amp_mx>85);   % we define a large peak is a peak that has larger than 85 amptitude;
        re_weight_gain =  (amp_mx(large_ind)./45);  % a large peak must have a higher weight as it can contain more than one particle. 
        w2(large_ind) = w2(large_ind).*re_weight_gain;
        w2 = w2./sum(w2);
    end
    answer_local = sortrows([f2, w2]);  % write the table as the sequence arrives. sort events along the time stamps. 
    data_Answer.EventID(k:k+N_answer-1) = eventID_wvf;
    data_Answer.ChannelID(k:k+N_answer-1) = ChanID_wvf;
    data_Answer.PETime(k:k+N_answer-1) = int16(answer_local(:,1));
    data_Answer.Weight(k:k+N_answer-1) = single(answer_local(:,2));
    k = k+N_answer;


    if mod(j,1000) ==0
        display(['current time =', datestr(now)]);
        display(['current progress =',num2str(j/M,3)]);
    end
 
end
toc

data_Answer.EventID = data_Answer.EventID(1:k-1);
data_Answer.ChannelID = data_Answer.ChannelID(1:k-1);
data_Answer.PETime  = data_Answer.PETime(1:k-1);
data_Answer.Weight = data_Answer.Weight(1:k-1);

awtable = struct2table(data_Answer);
writetable(awtable,output_file);
exit()
end

