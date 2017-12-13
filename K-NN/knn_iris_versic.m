% 2017 EC503 Project versic knn
%% load data
load('iris_versic_data.mat');
s = RandStream('mt19937ar','Seed',0);
rand = randperm(s, 150);
data = iris_versic_data(rand, 1:4);
label = iris_versic_data(rand, 5);
trainData = data(1:100, :);
trainLabel = label(1:100);
testData = data(101:150,:);
testLabel = label(101:150);

%% knn
%calculate distance between testdata points and traindata points
distance = pdist2(testData, trainData);
% TP FP FN TN
versic_rate = zeros(5,4);
%sort
[~, sortidx] = sort(distance,2);
%1nn
%get the closest one neighbor
one = trainLabel(sortidx(:,1));
%one nn confustion matrix
disp('1 nn confusion matrix');
oneconf = confusionmat(one, testLabel);
printmat(oneconf, 'Confusion Matrix', 'PredNormal PredAnomaly', 'GTNormal GTAnomaly');

versic_rate(1,1) = oneconf(2,2);
versic_rate(1,2) = oneconf(2,1);
versic_rate(1,3) = oneconf(1,2);
versic_rate(1,4) = oneconf(1,1);

%2nn
for i = 2:20
    fprintf('%d nn confusion matrix \n', i);
    twoidx = sortidx(:,1:i);
    two = mode(trainLabel(twoidx),2);
    twoconf = confusionmat(two, testLabel);
    printmat(twoconf, 'Confusion Matrix', 'PredNormal PredAnomaly', 'GTNormal GTAnomaly');
    versic_rate(i,1) = twoconf(2,2);
    versic_rate(i,2) = twoconf(2,1);
    versic_rate(i,3) = twoconf(1,2);
    versic_rate(i,4) = twoconf(1,1);
end

%% analyze
versic_precision = versic_rate(:,1)./(versic_rate(:,1) + versic_rate(:,2));
versic_recall = versic_rate(:,1)./(versic_rate(:,1) + versic_rate(:,3));
versic_fscore = 2*versic_precision.*versic_recall ./(versic_precision +versic_recall);

A = 1:20;
versic_ratetable = table(A.', versic_fscore, versic_precision, versic_recall);
versic_ratetable.Properties.VariableNames = {'knn', 'versic_fscore', 'versic_precision', 'versic_recall'}