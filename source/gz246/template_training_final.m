function [atom_dictionary] = template_training_final(scan_num, training_dataset_name)

if nargin < 1
    scan_num = 50000;
end

if nargin < 2 
    training_dataset_name = 'D:\proj_signalDetection\ftraining-6.h5';
end
filename = training_dataset_name;
fileinfo = h5info(filename );

datainfo = fileinfo.Datasets;
data_gt = h5read(filename,['/',datainfo(1).Name]);
data_trigger = h5read(filename,['/',datainfo(2).Name]);
data_waveform = h5read(filename,['/',datainfo(3).Name]);


% cd(path2);

waveform_record = nan(40,scan_num);

Scann_index = double(data_gt.PETime);
diff_scann_index = [Scann_index(1);abs(diff(Scann_index))];
diff2_scann_index = [abs(diff(Scann_index));Scann_index(end)];

select_set = logical(diff_scann_index>40)&logical(diff2_scann_index>40);
orginal_index = find((diff_scann_index>40)&(diff2_scann_index>40));


for ind = 1:scan_num
    gt_ind = orginal_index(ind);   % this is the index of ground truth;
    event_ind = double(data_gt.EventID(gt_ind));
    chan_ind = double(data_gt.ChannelID(gt_ind));
    petime_ind = double(data_gt.PETime(gt_ind));
    
    % we need to locate the waveform index via event, chan and petime id;
    A = find(data_waveform.EventID == event_ind);
    B = find(data_waveform.ChannelID(A) == chan_ind);
    try
        % de-mean:
        wf_ind = data_waveform.Waveform(petime_ind:(petime_ind+39),A(B)) - round(nanmean(data_waveform.Waveform(1:50,A(B))));
        waveform_record(:,ind) = wf_ind;
    catch
        waveform_record(:,ind) = nan;
    end
    
    if mod(ind,100) ==0
        display(['current progress =',num2str(ind/scan_num)]);
    end
    
end

peaks_dist = min(waveform_record,[],1);
% figure();hist(peaks_dist,200);

waveform_rec_adj  = waveform_record./repmat(abs(peaks_dist),40,1);
waveform_rec_adj(isinf(waveform_rec_adj)) = nan;


fm = nanmean(waveform_rec_adj,2);  % default format; 

fm2 = fm;

fm3 = smooth(fm2,'lowess');


fm_new = [fm(1:19);fm3(20:end)];
fm_new = fm_new./sqrt(sum(fm_new.^2));


atom_dic2 = zeros(1029,1029);
[m,n] = size(atom_dic2);
for i = 1:(1029-39)
    atom_dic2(i:i+39,i) = fm_new;
end
% new version to cover a little bit more time span
atom_dic3 = zeros(1029,1029);
[m,n] = size(atom_dic3);
for i = 1:n
    if i<=980
        atom_dic3(:,i) = atom_dic2(:,i);
    elseif i+39<=1029
        atom_dic3(i:i+39,i) = atom_dic2(1:40,1);
    else
        atom_dic3(i:end,i) = atom_dic2(1:(m-i+1),1);
    end
end

atom_dictionary = atom_dic3;

figure();plot(fm_new);grid();
title('unit impluse response extracted');
save('atom_dictionary.mat','atom_dictionary');

end

