% 2017 EC503 Project sanota svm
%% load data
load('iris_sanota_data.mat');
s = RandStream('mt19937ar','Seed',0);
rand = randperm(s, 150);
data = iris_sanota_data(rand, 1:4);
label = iris_sanota_data(rand, 5);
trainData = data(label == 1,:);
trainLabel(1:size(trainData,1),1) = 1; %1 normal
testlabel(label == 1,1) = 1;
testlabel(label ~= 1,1) = -1;

%% SVM
%set nu range
m(1,:) = '-s 2 -t 2 -n 0.001';
m(2,:) = '-s 2 -t 2 -n 0.002';
m(3,:) = '-s 2 -t 2 -n 0.004';
m(4,:) = '-s 2 -t 2 -n 0.006';
m(5,:) = '-s 2 -t 2 -n 0.008';
m(6,:) = '-s 2 -t 2 -n 0.010';
m(7,:) = '-s 2 -t 2 -n 0.012';
m(8,:) = '-s 2 -t 2 -n 0.014';
m(9,:) = '-s 2 -t 2 -n 0.016';
m(10,:) = '-s 2 -t 2 -n 0.018';
n = 0.000:0.002:0.018;

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
xlim([0.0 0.018]);
xlabel('nu');
ylabel('value');
title('iris satona with OCSVM');
legend('precision', 'recall', 'fscore');