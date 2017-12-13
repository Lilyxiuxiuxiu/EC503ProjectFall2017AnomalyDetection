% 2017 EC503 Project pen-global knn
%% load data
load('pg_data.mat');
s = RandStream('mt19937ar','Seed',0);
rand = randperm(s, 809);
label = pg_data(rand,17);
data = pg_data(rand,1:16);
%normal 1 abnormal 2
trainData = data(label == 1, :);
trainLabel(1:size(trainData,1),1) = 1; %1 normal
testlabel(label == 1,1) = 1; %normal
testlabel(label ~= 1,1) = -1; %anomaly

%% SVM
%set nu range
m(1,:) = '-s 2 -t 2 -n 0.89999900';
m(2,:) = '-s 2 -t 2 -n 0.89999990';
m(3,:) = '-s 2 -t 2 -n 0.90000000';
m(4,:) = '-s 2 -t 2 -n 0.99000000';
m(5,:) = '-s 2 -t 2 -n 0.99900000';
m(6,:) = '-s 2 -t 2 -n 0.99990000';
m(7,:) = '-s 2 -t 2 -n 0.99999000';
m(8,:) = '-s 2 -t 2 -n 0.99999900';
m(9,:) = '-s 2 -t 2 -n 0.99999990';
m(10,:) = '-s 2 -t 2 -n 0.99999999';
n = [0.89999900; 0.89999990; 0.90000000; 0.99000000; 0.99900000;...
    0.99990000; 0.99999000; 0.99999900; 0.99999990; 0.99999999];

for i = 1:10
    %train
    model = svmtrain(trainLabel, trainData, m(i,:));
    %test
    [predicted_label(i,:), accuracy, decision_values] = svmpredict(testlabel, data, model);
    %p= -predicted_label;
    confmat = confusionmat(predicted_label(i,:), testlabel);
    printmat(confmat, 'Confusion Matrix of OCSVM for pg_data', 'PredAnomaly PredNormal', 'GTAnomaly GTNormal');
    TP(i) = confmat(1,1);
    FP(i) = confmat(1,2);
    FN(i) = confmat(2,1);
    TN(i) = confmat(2,2);
end

%% analyze
precision = TP./(TP+FP);
recall = TP./(TP+FN);
fscore = 2.*precision.*recall./(precision+recall);

figure
plot(n, precision);
hold on
plot(n, recall);
plot(n, fscore);
xlim([0.89999900 1]);
xlabel('nu');
ylabel('value');
title('Pen-Global with OCSVM');
legend('precision', 'recall', 'fscore');