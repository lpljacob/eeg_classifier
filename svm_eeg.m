% Input data: one three-dimensional array file per subject named subdata[number].mat
% Array consists of electrodes x timepoints (one voltage value per electrode per timepoint) x trials.
% Each subject also needs a separate vector with trial info named trialinfo[number].mat
% Vector length must be equal to third dimension of subdata

% -------------------------------------------------------------------------    
%                                  SET UP
% -------------------------------------------------------------------------

n = 20; % number of subjects
folder = 'D:\Experiment data\ERP Priming\Word Priming Delayed\svm files\'; % trial data location
f_save = 'C:\Users\UMass\OneDrive\_zGrad\_ERP PRIMING PROJECT\SVM\results\';
timewind_size = 50;
electrodes = 64;
conds = 8; % number of conditions
setaside = 5; % trials per condition per subject to set aside for testing

version = 2;
% 1 - all time bins
% 2 - individual time bins

save_weights = 1; % wether or not to save weight vectors

% preallocate space for data
alldata = nan(896,500,20);
alltrialinfo = nan(1,500,20);

% experiment specific: condition coding
condition_pairs = [13 14 15 16 9 10 11 12];

% -------------------------------------------------------------------------    
%                             PREPROCESSING
% -------------------------------------------------------------------------

for i = 1:n
    
    % load subject data and set up variables
    load([folder sprintf('subdata%d.mat',i)]);
    timewinds = size(datasave,2)/timewind_size; 
    data = zeros(size(datasave,1), timewinds, size(datasave,3));
    
    % acquire average of each time window
    for j=1:timewinds
        data(:, j, :) = mean(datasave(:,(j*timewind_size)-(timewind_size-1):(j*timewind_size),:),2);
    end
    
    % normalize data and reshape each trial into a vector
    normdata = zeros(size(data));
    for j=1:size(data,3)
        datatmp = data(:,:,j);
        normdata(:, :, j) = (data(:, :, j) - mean(datatmp(:))) / std(datatmp(:));
    end
    
    reshaped_data = reshape(normdata, size(data,1)*size(data,2), size(data,3));
    
    % load trial info and store subject data
    load([folder sprintf('trialinfo%d.mat',i)]);
    
    alldata(:, 1:size(reshaped_data, 2), i) = reshaped_data;
    alltrialinfo(:, 1:size(trialsave, 2), i) = trialsave;
    
end

% get labels of interest
all_labels = nan(size(alltrialinfo));

% experiment specific
all_labels(alltrialinfo<9)=-1; % Same choice
all_labels(alltrialinfo>8)=+1; % Different choice

% -------------------------------------------------------------------------    
%                           SET ASIDE TEST DATA
% -------------------------------------------------------------------------

% TRAINING AND TESTING LOOP
% samples data multiple times for reliability

loops = 1000;
if version == 1
    vector_size = electrodes*timewinds;
elseif version == 2
    vector_size = electrodes;
    meanacc = nan(1, timewinds);
end

subsacc = nan(1,n, loops);
condacc = nan(1,conds, loops);
allacc = nan(1, loops);
allweights = nan(vector_size, loops);


for tp = 1:timewinds  % only run the outer loop when doing version 2
    
    tp_start = (tp-1)*electrodes+1;
    tp_end = tp_start+electrodes-1;
    
    for k = 1:loops

        % preallocate space
        traindata = [];
        trainlabels = [];
        testdata = nan(vector_size, 5, 8, 20);
        testlabels = nan(1, 5, 8, 20);

        rng('shuffle')
        for i = 1:n
            % obtain data for this subject
            subdata = alldata(:,~isnan(alldata(1, :, i)),i);
            subinfo = alltrialinfo(:,~isnan(alltrialinfo(1, :, i)),i);
            sublabels = all_labels(:,~isnan(all_labels(1, :, i)),i);

            % randomly sample from each condition and add rest to traindata
            for j = 1:conds
                condata = subdata(:,subinfo==j | subinfo==condition_pairs(j));
                conlabels = sublabels(:,subinfo==j | subinfo==condition_pairs(j));

                idx = randperm(size(condata,2));
                
                if version==1
                    testdata(:, :, j, i) = condata(:,idx(1:5));
                    traindata = [traindata, condata(:,idx(6:end))];
                elseif version==2
                    testdata(:, :, j, i) = condata(tp_start:tp_end,idx(1:5));
                    traindata = [traindata, condata(tp_start:tp_end,idx(6:end))];
                end
                
                testlabels(:, :, j, i) = conlabels(:,idx(1:5));
                trainlabels = [trainlabels, conlabels(:,idx(6:end))];     
            end
        end


        % TRAIN MODEL

        model = svmtrain2(trainlabels', traindata');

        % TEST MODEL

        if version == 1
            % subjects accuracy
            for i = 1:n
                tmptest = testdata(:, :, :, i);
                tmptest = reshape(tmptest, vector_size, 40);

                tmplabels = testlabels(:, :, :, i);
                tmplabels = reshape(tmplabels, 1, 40);

                [~, accuracy, ~] = svmpredict(tmplabels', tmptest', model);
                subsacc(:, i, k) = accuracy(1);
            end

            % condition accuracy
            for i = 1:conds
                tmptest = testdata(:, :, i, :);
                tmptest = reshape(tmptest, vector_size, 100);

                tmplabels = testlabels(:, :, i, :);
                tmplabels = reshape(tmplabels, 1, 100);

                [~, accuracy, ~] = svmpredict(tmplabels', tmptest', model);
                condacc(:, i, k) = accuracy(1);
            end
        end

        % overall accuracy
        tmptest = reshape(testdata, vector_size, 800);
        tmplabels = reshape(testlabels, 1, 800);

        [~, accuracy, ~] = svmpredict(tmplabels', tmptest', model);
        allacc(:, k) = accuracy(1);

        if save_weights == 1
            wts = model.SVs' * model.sv_coef;
            allweights(:, k) = wts;
        end
        
        if version == 2
            fprintf('loop %d of timewindow %d done \n', k, tp)
        else
            fprintf('loop %d done \n', k)
        end

    end

if save_weights == 1
    meanwts = mean(allweights, 2);
    save([f_save sprintf('weights_time%d.mat',tp)], 'meanwts');
end

meanacc(1, tp) = mean(allacc, 2);
end

save([f_save 'acc.mat'], 'meanacc');

% save version 1 results
meansubacc = mean(subsacc, 3);
save([f_save 'subs.mat'], 'meansubacc');
meancondacc = mean(condacc, 3);
save([f_save 'conds.mat'], 'meancondacc');
meanacc = mean(allacc, 2);
save([f_save 'acc.mat'], 'meanacc');
meanwts = mean(allweights, 2);
save([f_save 'weights.mat'], 'meanwts');
