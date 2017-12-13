% 2017 EC503 Project sanota knn
%% load data
load('iris_sanota_data.mat');
s = RandStream('mt19937ar','Seed',0);
rand = randperm(s, 150);
data = iris_sanota_data(rand, 1:4);
label = iris_sanota_data(rand, 5);
trainData = data(1:100, :);
trainLabel = label(1:100);
testData = data(101:150,:);
testLabel = label(101:150);

%% knn
%calculate distance between testdata points and traindata points
distance = pdist2(testData, trainData);
% TP FP FN TN
sanota_rate = zeros(5,4);
%sort
[~, sortidx] = sort(distance,2);
%1nn
%get the closest one neighbor
one = trainLabel(sortidx(:,1));
%one nn confustion matrix
disp('1 nn confusion matrix');
oneconf = confusionmat(one, testLabel);
printmat(oneconf, 'Confusion Matrix', 'PredNormal PredAnomaly', 'GTNormal GTAnomaly');

sanota_rate(1,1) = oneconf(2,2);
sanota_rate(1,2) = oneconf(2,1);
sanota_rate(1,3) = oneconf(1,2);
sanota_rate(1,4) = oneconf(1,1);

%2nn
for i = 2:20
    fprintf('%d nn confusion matrix \n', i);
    twoidx = sortidx(:,1:i);
    two = mode(trainLabel(twoidx),2);
    twoconf = confusionmat(two, testLabel);
    printmat(twoconf, 'Confusion Matrix', 'PredNormal PredAnomaly', 'GTNormal GTAnomaly');
    sanota_rate(i,1) = twoconf(2,2);
    sanota_rate(i,2) = twoconf(2,1);
    sanota_rate(i,3) = twoconf(1,2);
    sanota_rate(i,4) = twoconf(1,1);
end

%% analyze
sanota_precision = sanota_rate(:,1)./(sanota_rate(:,1) + sanota_rate(:,2));
sanota_recall = sanota_rate(:,1)./(sanota_rate(:,1) + sanota_rate(:,3));
sanota_fscore = 2*sanota_precision.*sanota_recall ./(sanota_precision +sanota_recall);

A = 1:20;
sanota_ratetable = table(A.', sanota_fscore, sanota_precision, sanota_recall);
sanota_ratetable.Properties.VariableNames = {'knn', 'sanota_fscore', 'sanota_precision', 'sanota_recall'}