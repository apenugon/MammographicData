function [GenErr RandErr] = DHM2()
rng(47);
data = csvread('mammographic_masses.data');
newData = [];
for i = 1:size(data, 1),
    curRow = data(i,:);
    if size(curRow(curRow == -1), 2) == 0,
        newData = [newData; curRow];
    end
end
newData
size(newData)
featureData = data(:,2:5);
TRUE_LABELS = data(:,6);
numsamples = 100;
Sup = zeros(numsamples, 1);
S = zeros(numsamples, 1);
SLabels = zeros(numsamples, 1);
T = zeros(numsamples, 1);
TLabels = zeros(numsamples,1);
SUnionT = zeros(numsamples, 1);
SUnionTLabels = zeros(numsamples, 1);

random_indicies = randsample(numsamples,numsamples);
cost = 0;
costcurve=zeros(1, numsamples);
GenErr = sum(TRUE_LABELS)/numsamples;
SupErr = sum(TRUE_LABELS)/numsamples;
upper = 1;
cursvm = fitcsvm(featureData(random_indicies(1:upper),:),TRUE_LABELS(random_indicies(1:upper)));
T(random_indicies(1:upper)) = 1;
TLabels(random_indicies(1:upper)) = TRUE_LABELS(random_indicies(1:upper));
SUnionT = T;
SUnionTLabels = TLabels;
sup(random_indicies(1:upper)) = 1;
precision = costcurve;
recall = costcurve;
f1score = costcurve;
for t = upper+1:numsamples
    t
    index = random_indicies(t);
    x_t = featureData(index,:);
    sup(index) = 1;
    supsvm = fitcsvm(featureData(sup==1,:), TRUE_LABELS(sup==1));
    supPreds = predict(supsvm, featureData);
    SupErr(t) = sum(supPreds ~= TRUE_LABELS)/size(featureData, 1);
    
    [svmPlus, flagPlus] = learn(featureData(SUnionT==1,:),SUnionTLabels(SUnionT==1), x_t, 1, cursvm);
    [svmMinus, flagMinus] = learn(featureData(SUnionT==1,:),SUnionTLabels(SUnionT==1), x_t, 0, cursvm);
    
    hpluserr = getErr(svmPlus, featureData(SUnionT==1,:), SUnionTLabels(SUnionT==1));
    hminuserr = getErr(svmMinus, featureData(SUnionT==1,:), SUnionTLabels(SUnionT==1));
    hpluserr-hminuserr;
    beta = .1*sqrt((5*log(t)+log(1/0.05))/t);
    Delta = (beta^2 + beta*(sqrt(hpluserr)+sqrt(hminuserr)));
    
    
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
        cursvm = fitcsvm(featureData(SUnionT == 1, :), SUnionTLabels(SUnionT == 1));
        cost = cost + 1;
        sPreds = predict(cursvm, featureData);
        GenErr(cost) = sum(sPreds ~= TRUE_LABELS)/size(featureData, 1);
    end
    costcurve(t) = cost;
    
    curPredictions = predict(cursvm, featureData);
    numerator = size(find(curPredictions == 1 & TRUE_LABELS == 1), 1)
    denom = sum(TRUE_LABELS == 1);
    recall(t) = numerator/denom;
    
    denom = sum(curPredictions==1);
    precision(t) = numerator/denom;
    
end
figure(1);
plot(costcurve);
figure(2);
plot(GenErr, 'b');
hold on;
plot(SupErr, 'r');
hold off;

f1score = 2 * ((precision.*recall)./(precision+recall));

figure(3);
plot(precision, 'b');
hold on;
plot(recall, 'r');
hold on;
plot(f1score, 'g');
hold off;


end

function [err] = getErr(svm, features, labels)
    data = predict(svm, features);
    err = sum(data ~= labels)/numel(data);
end

function [svm, flag] = learn(training_data, training_labels, new_point,assigned_label, old_svm)
    svm = fitcsvm([training_data;new_point], [training_labels;assigned_label]);
    % consistency check
    
    if isempty(training_data)
       flag = 0;
       'Empty Training Data' 
       return;
    end
    
    newErr = getErr(svm, [training_data;new_point], [training_labels;assigned_label]);
    oldErr = getErr(old_svm, [training_data;new_point], [training_labels;assigned_label]);
    flag = newErr>oldErr;
end