function [] = DHM()

data = csvread('mammographic_masses.data');
featureData = data(:,2:5);
TRUE_LABELS = data(:,6);
numsamples = 200;
S = zeros(numsamples, 1);
SLabels = zeros(numsamples, 1);
T = zeros(numsamples, 1);
TLabels = zeros(numsamples,1);
SUnionT = zeros(numsamples, 1);
SUnionTLabels = zeros(numsamples, 1);

random_indicies = randsample(numsamples,numsamples);
prevDataSampled = [];
cursvm = fitcsvm([1], [1], 'ClassNames', [0, 1]);
cost = 0;
costcurve=zeros(1, numsamples);
GenErr = sum(TRUE_LABELS)/numsamples;
generalizationError = [];
for t = 1:numsamples
    t
    %index = random_indicies(t);
    index = t;
    x_t = featureData(index,:);
    
    [svmPlus, flagPlus] = learn(featureData(SUnionT==1,:),TRUE_LABELS(SUnionT==1), x_t, 1, cursvm);
    [svmMinus, flagMinus] = learn(featureData(SUnionT==1,:),TRUE_LABELS(SUnionT==1), x_t, 0, cursvm);
    
    hpluserr = getErr();
    hminuserr = getErr();
    
    Delta = inf;
    
    if flagPlus == 1
        % positive case failed
        S(index) = 1;
        SUnionT(index) = 1;
        SLabels(index) = 0;
        SUnionTLabels(index) = 0;
        cursvm = svmMinus;
    elseif flagMinus == 1
        % negative case failed
        S(index) = 1;
        SUnionT(index) = 1;
        SLabels(index) = 1;
        SUnionTLabels(index) = 1;
        cursvm = svmPlus;
    elseif (hminuserr-hpluserr) > Delta
        % Add positive case
        S(index) = 1;
        SUnionT(index) = 1;
        SLabels(index) = 1;
        SUnionTLabels(index) = 1;
        cursvm = svmPlus;
    elseif (hpluserr-hminuserr) > Delta
        S(index) = 1;
        SUnionT(index) = 1;
        SLabels(index) = 0;
        SUnionTLabels(index) = 0;
        cursvm = svmMinus;
    else
        T(index) = 1;
        SUnionT(index) = 1;
        TLabels(index) = TRUE_LABELS(index);
        SUnionTLabels(index) = TLabels(index);
        %TODO: Assign new SVM correctly
        cursvm = fitcsvm([featureData(SUnionT == 1, :);x_t], [SUnionTLabels(SUnionT == 1);TRUE_LABELS(index)], 'ClassNames', [0, 1]);
        cost = cost + 1
        sPreds = predict(cursvm, featureData);
        GenErr(cost) = sum(sPreds ~= TRUE_LABELS)/size(featureData, 1);
    end
        
    costcurve(t) = cost;
    
end
figure(1);
plot(costcurve);
figure(2);
plot(GenErr);

end

function [err] = getErr()
    err = 0;
end

function [svm, flag] = learn(training_data, training_labels, new_point,assigned_label, old_svm)
    svm = fitcsvm([training_data;new_point], [training_labels;assigned_label], 'ClassNames', [0, 1]);
    % consistency check
    
    if isempty(training_data)
       flag = 0;
       'Empty Training Data' 
       return;
    end
    
    numSupportVectorsNew = size(svm.SupportVectors, 1);
    numSupportVectorsOld = size(old_svm.SupportVectors, 1);
    diff = abs(numSupportVectorsNew - numSupportVectorsOld);
    
    if diff < 2,
       flag = 0; 
    else,
       flag = 1;
    end
    
    %[newPredictions newScore] = predict(svm, [training_data; new_point]);
    %[oldPredictions oldScore] = predict(old_svm, [training_data; new_point]);
    %newScore
    %oldScore
    %flag = newScore > oldScore;
    
    %flag = not(isempty(find(newPredictions == training_labels) == 0));
end