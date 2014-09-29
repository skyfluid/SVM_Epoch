clc
clear
close all

def_unit_count = 16;
def_sampling_ratio = 50;

PRE = load(['../Replay/data/PRE_winArr_' int2str(def_unit_count) 'u_50bs.mat']);
TASK = load(['../Replay/data/TASK_winArr_' int2str(def_unit_count) 'u_50bs.mat']);
POST = load(['../Replay/data/POST_winArr_' int2str(def_unit_count) 'u_50bs.mat']);

length_PRE = length(PRE.stampSet);
length_TASK = length(TASK.stampSet);
length_POST = length(POST.stampSet);

PRE.stampSet = PRE.stampSet((length_PRE/3):length_PRE, :);
length_PRE = length(PRE.stampSet);

trialSet = [PRE.stampSet; TASK.stampSet; POST.stampSet];
ansSet = [repmat(1, length_PRE, 1); repmat(2, length_TASK, 1); repmat(3, length_POST, 1)];

% random permute trialSet/ansSet
n = size(trialSet,1);
p = randperm(n);
trialSet = trialSet(p,:);
ansSet = ansSet(p,:);

trialSet = trialSet(1:(length(trialSet)/def_sampling_ratio),:);
ansSet = ansSet(1:(length(ansSet)/def_sampling_ratio),:);

clear PRE;
clear TASK;
clear POST;     

stat_total = {};

%for nPC=1:50
    %stat_total.data(nPC).HRtest = [];
    %stat_total.data(nPC).HRWhole = [];
    stat_total.data.HRtest = [];
    stat_total.data.HRWhole = [];
    
    SVM_candidate = [];
    
    for iTestSet=1:10
        testBegin = (10-iTestSet)*(length(ansSet)/10)+1;
        testEnd = (11-iTestSet)*(length(ansSet)/10);
        testSet = trialSet(testBegin:testEnd,:);
        %testSetw = wordSet(testBegin:testEnd,:);
        testSetAns = ansSet(testBegin:testEnd,:);
        trainingSet = [trialSet(1:testBegin-1,:); trialSet(testEnd+1:end,:)];
        %trainingSetw = [wordSet(1:testBegin-1,:); wordSet(testEnd+1:end,:)];
        trainingSetAns = [ansSet(1:testBegin-1,:); ansSet(testEnd+1:end,:)];

        local_best_score = 0;
        
        for iWorkSet=1:9
            verifyBegin = (9-iWorkSet)*(length(trainingSet)/9)+1;
            verifyEnd = (10-iWorkSet)*(length(trainingSet)/9);
            verifySet = trainingSet(verifyBegin:verifyEnd,:);
            %verifySetw = trainingSetw(verifyBegin:verifyEnd,:);
            verifySetAns = trainingSetAns(verifyBegin:verifyEnd,:);
            workSet = [trainingSet(1:verifyBegin-1,:); trainingSet(verifyEnd+1:end,:)];
            %workSetw = [trainingSetw(1:verifyBegin-1,:); trainingSetw(verifyEnd+1:end,:)];
            workSetAns = [trainingSetAns(1:verifyBegin-1,:); trainingSetAns(verifyEnd+1:end,:)];

            %[coeff, score, latent] = pca(workSet, 'Centered', true);  % PCA

            %varCov = cumsum(latent)./sum(latent);    % find the variance coverage for PCs
            %nPC = find(varCov > VarCovThreshold, 1);
            %scoreSel = (workSet) * coeff(:, 1:nPC);
            scoreSel = (workSet);
            
            %%% libSVM
 %           options.MaxIter 
            %svmStruct = svmtrain(scoreSel, distingSet); %, 'showplot',true);
            svmStruct = svmtrain(workSetAns, scoreSel); %, 'showplot',true);
            %resultSet = svmclassify(svmStruct, verifySet*coeff(:, 1:nPC)); %, 'showplot',true);
            %tSet = verifySet*coeff(:, 1:nPC);
            tSet = verifySet;
            %tSetWhole = trialSet*coeff(:, 1:nPC);
            tSetWhole = trialSet;
            [resultSet, accuracy, decision_values] = svmpredict(verifySetAns, tSet, svmStruct); %, 'showplot',true);
            %decision_values
            
            if (accuracy(1) > local_best_score)
                local_best_score = accuracy(1);
                SVM_candidate{iTestSet}.svm = svmStruct;
                %SVM_candidate{iTestSet}.pc = coeff(:, 1:nPC);
                %pcvar = (cumsum(latent) ./ sum(latent));
                %SVM_candidate{iTestSet}.pcvar = pcvar(nPC);
            end
        end
        %[resultSet, accuracy, decision_values] = svmpredict(testSetAns, testSet*SVM_candidate{iTestSet}.pc, SVM_candidate{iTestSet}.svm);
        [resultSet, accuracy, decision_values] = svmpredict(testSetAns, testSet, SVM_candidate{iTestSet}.svm);
        %[resultSetWhole, accuracyWhole, decision_valuesWhole] = svmpredict(ansSet, tSetWhole*SVM_candidate{iTestSet}.pc, svmStruct); %, 'showplot',true);
        [resultSetWhole, accuracyWhole, decision_valuesWhole] = svmpredict(ansSet, tSetWhole, svmStruct); %, 'showplot',true);
        
        %fprintf(outfile, 'SVM-%2d, #TestVec: %d, #HR: %f, #PC:%d\n', iTestSet, length(resultSet), accuracy(1), nPC);
                
        stat_total.data.HRtest = [stat_total.data.HRtest accuracy];
        stat_total.data.HRtestPRE(iTestSet)  = sum(resultSet(find(testSetAns == 1)) == testSetAns(find(testSetAns == 1)) ) / length(find(testSetAns));
        stat_total.data.HRtestTASK(iTestSet) = sum(resultSet(find(testSetAns == 2)) == testSetAns(find(testSetAns == 2)) ) / length(find(testSetAns));
        stat_total.data.HRtestPOST(iTestSet) = sum(resultSet(find(testSetAns == 3)) == testSetAns(find(testSetAns == 3)) ) / length(find(testSetAns));
        %stat_total.data.PCtestvar(iTestSet) = SVM_candidate{iTestSet}.pcvar;
        stat_total.data.HRWhole = [stat_total.data.HRWhole accuracyWhole];
        stat_total.data.HRWholePRE(iTestSet)  = sum(resultSetWhole(find(ansSet == 1)) == ansSet(find(ansSet == 1)) ) / length(find(ansSet));
        stat_total.data.HRWholeTASK(iTestSet) = sum(resultSetWhole(find(ansSet == 2)) == ansSet(find(ansSet == 2)) ) / length(find(ansSet));
        stat_total.data.HRWholePOST(iTestSet) = sum(resultSetWhole(find(ansSet == 3)) == ansSet(find(ansSet == 3)) ) / length(find(ansSet));
        
        
    end
    stat_total.avgHRtest       = mean(stat_total.data.HRtest(1,:), 2);
    stat_total.avgHRtestPRE    = mean(stat_total.data.HRtestPRE, 2);
    stat_total.avgHRtestTASK   = mean(stat_total.data.HRtestTASK, 2);
    stat_total.avgHRtestPOST   = mean(stat_total.data.HRtestPOST, 2);
    %stat_total.avgPCtestvar    = mean(stat_total.data.PCtestvar, 2);
    stat_total.avgHRWhole       = mean(stat_total.data.HRWhole(1,:), 2);
    stat_total.avgHRWholePRE    = mean(stat_total.data.HRWholePRE, 2);
    stat_total.avgHRWholeTASK   = mean(stat_total.data.HRWholeTASK, 2);
    stat_total.avgHRWholePOST   = mean(stat_total.data.HRWholePOST, 2);
    %stat_total.avgPCWholevar    = mean(stat_total.data.PCWholevar, 2);
    
    save('./results/CrossValid_Epoch.mat', 'stat_total')
%end

%fclose(outfile);


save('./results/CrossValid_Epoch.mat', 'stat_total')

set(0, 'DefaultFigureVisible', 'on')
