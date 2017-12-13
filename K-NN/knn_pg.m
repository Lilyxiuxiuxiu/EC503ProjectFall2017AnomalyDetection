% 2017 EC503 Project pen-global knn
%% load data
load('pg_data.mat');
s = RandStream('mt19937ar','Seed',0);
rand = randperm(s, 809);
label = pg_data(rand,17);
data = pg_data(rand,1:16);
trainData = data(rand(1:600),:);
trainLabel = label(rand(1:600),:);
testData = data(rand(601:809),:);
testLabel = label(rand(601:809),:);

%% knn
%calculate distance between testdata points and traindata points
distance = pdist2(testData, trainData);
% TP FP FN TN
pg_rate = zeros(20,4);
%sort
[~, sortidx] = sort(distance,2);
%1nn
%get the closest one neighbor
one = trainLabel(sortidx(:,1));
%one nn confustion matrix
disp('1 nn confusion matrix');
oneconf = confusionmat(one, testLabel);
printmat(oneconf, 'Confusion Matrix', 'PredNormal PredAnomaly', 'GTNormal GTAnomaly');

pg_rate(1,1) = oneconf(2,2);
pg_rate(1,2) = oneconf(2,1);
pg_rate(1,3) = oneconf(1,2);
pg_rate(1,4) = oneconf(1,1);

%2nn
for i = 2:20
    fprintf('%d nn confusion matrix \n', i);
    twoidx = sortidx(:,1:i);
    two = mode(trainLabel(twoidx),2);
    twoconf = confusionmat(two, testLabel);
    printmat(twoconf, 'Confusion Matrix', 'PredNormal PredAnomaly', 'GTNormal GTAnomaly');
    pg_rate(i,1) = twoconf(2,2);
    pg_rate(i,2) = twoconf(2,1);
    pg_rate(i,3) = twoconf(1,2);
    pg_rate(i,4) = twoconf(1,1);
end


pg_precision = pg_rate(:,1)./(pg_rate(:,1) + pg_rate(:,2));
pg_recall = pg_rate(:,1)./(pg_rate(:,1) + pg_rate(:,3));
pg_fsocre = 2*pg_precision.*pg_recall ./(pg_precision +pg_recall);

A = 1:20;
pg_ratetable = table(A.', pg_fsocre, pg_precision, pg_recall);
pg_ratetable.Properties.VariableNames = {'knn', 'pg_fsocre', 'pg_precision', 'pg_recall'}