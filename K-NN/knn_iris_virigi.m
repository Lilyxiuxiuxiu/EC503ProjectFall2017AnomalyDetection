% 2017 EC503 Project virigi knn
%% load data
load('iris_virigi_data.mat');
s = RandStream('mt19937ar','Seed',0);
rand = randperm(s, 150);
data = iris_virigi_data(rand, 1:4);
label = iris_virigi_data(rand, 5);
trainData = data(1:100, :);
trainLabel = label(1:100);
testData = data(101:150,:);
testLabel = label(101:150);

%% knn
%calculate distance between testdata points and traindata points
distance = pdist2(testData, trainData);
% TP FP FN TN
virigi_rate = zeros(5,4);
%sort
[~, sortidx] = sort(distance,2);
%1nn
%get the closest one neighbor
one = trainLabel(sortidx(:,1));
%one nn confustion matrix
disp('1 nn confusion matrix');
oneconf = confusionmat(one, testLabel);
printmat(oneconf, 'Confusion Matrix', 'PredAnomaly PredNormal', 'GTAnomaly GTNormal');

virigi_rate(1,1) = oneconf(1,1);
virigi_rate(1,2) = oneconf(1,2);
virigi_rate(1,3) = oneconf(2,1);
virigi_rate(1,4) = oneconf(2,2);

%2nn
for i = 2:20
    fprintf('%d nn confusion matrix \n', i);
    twoidx = sortidx(:,1:i);
    two = mode(trainLabel(twoidx),2);
    twoconf = confusionmat(two, testLabel);
    printmat(twoconf, 'Confusion Matrix', 'PredAnomaly PredNormal', 'GTAnomaly GTNormal');
    virigi_rate(i,1) = twoconf(1,1);
    virigi_rate(i,2) = twoconf(1,2);
    virigi_rate(i,3) = twoconf(2,1);
    virigi_rate(i,4) = twoconf(2,2);
end

%% analyze
virigi_precision = virigi_rate(:,1)./(virigi_rate(:,1) + virigi_rate(:,2));
virigi_recall = virigi_rate(:,1)./(virigi_rate(:,1) + virigi_rate(:,3));
virigi_fscore = 2*virigi_precision.*virigi_recall ./(virigi_precision +virigi_recall);

A = 1:20;
virigi_ratetable = table(A.', virigi_fscore, virigi_precision, virigi_recall);
virigi_ratetable.Properties.VariableNames = {'knn', 'virigi_fscore', 'virigi_precision', 'virigi_recall'}