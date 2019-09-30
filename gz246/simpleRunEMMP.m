output_file='MyAnswer.txt';
training_file='../../killxbq/training.h5';
% [atom_dictionary] = template_training_final(5000,training_file);
% atom_dictionary need load
load atom_dictionary
% data_waveform = h5read(target_file,['/','Waveform']);
data_waveform = WaveformGroup;
M = length(data_waveform.EventID);

data_Answer.EventID = int64(nan(M.*30,1));
data_Answer.ChannelID = int16(nan(M.*30,1));
data_Answer.PETime = int16(nan(M.*30,1));
data_Answer.Weight = single(nan(M.*30,1));
% preallocate 30PE,maybe not enough,but accelerate the process
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
   % calculate the weight
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
