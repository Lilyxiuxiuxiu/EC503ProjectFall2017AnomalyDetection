% 2017 EC503 Project kdd knn
%% load data
load('kddData.mat');
s = RandStream('mt19937ar','Seed',0);
%change anamoly size
normalkdd = kddData(kddData(:,42) == 0,:);
anakddinx = find(kddData(:,42) == 1);
rand1 = randperm(s, size(anakddinx,1));
selectinx = anakddinx(rand1(1:1000));
%make smaller data set
newkddData = [normalkdd;kddData(selectinx,:)];
random=randperm(s, size(newkddData,1));
kddDatasmall = newkddData(random(1:20000),:);
random2 = randperm(s, size(kddDatasmall, 1));
%seperare train and test
train = kddDatasmall(random2(1:15000),:);
test = kddDatasmall(random2(15001:20000),:);
trainData = train(:,1:41);
trainLabel = train(:,42);
testData = test(:,1:41);
testLabel = test(:,42);

%% knn
%calculate distance between testdata points and traindata points
distance = pdist2(testData, trainData);
% TP FP FN TN
kdd_rate = zeros(5,4);
%sort
[~, sortidx] = sort(distance,2);
%1nn
%get the closest one neighbor
one = trainLabel(sortidx(:,1));
%one nn confustion matrix
disp('1 nn confusion matrix');
oneconf = confusionmat(one, testLabel);
printmat(oneconf, 'Confusion Matrix', 'PredNormal PredAnomaly', 'GTNormal GTAnomaly');

kdd_rate(1,1) = oneconf(2,2);
kdd_rate(1,2) = oneconf(2,1);
kdd_rate(1,3) = oneconf(1,2);
kdd_rate(1,4) = oneconf(1,1);

%2nn
for i = 2:20
    fprintf('%d nn confusion matrix \n', i);
    twoidx = sortidx(:,1:i);
    two = mode(trainLabel(twoidx),2);
    twoconf = confusionmat(two, testLabel);
    printmat(twoconf, 'Confusion Matrix', 'PredNormal PredAnomaly', 'GTNormal GTAnomaly');
    kdd_rate(i,1) = twoconf(2,2);
    kdd_rate(i,2) = twoconf(2,1);
    kdd_rate(i,3) = twoconf(1,2);
    kdd_rate(i,4) = twoconf(1,1);
end

%% analyze
kdd_precision = kdd_rate(:,1)./(kdd_rate(:,1) + kdd_rate(:,2));
kdd_recall = kdd_rate(:,1)./(kdd_rate(:,1) + kdd_rate(:,3));
kdd_fscore = 2*kdd_precision.*kdd_recall ./(kdd_precision +kdd_recall);

A = 1:20;
kdd_ratetable = table(A.', kdd_fscore, kdd_precision, kdd_recall);
kdd_ratetable.Properties.VariableNames = {'knn', 'kdd_fscore', 'kdd_precision', 'kdd_recall'}