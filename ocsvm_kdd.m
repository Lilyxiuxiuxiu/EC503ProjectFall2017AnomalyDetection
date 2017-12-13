% 2017 EC503 Project versic svm
%% load data
load('kddData.mat');
s = RandStream('mt19937ar','Seed',0);
normalkdd = kddData(kddData(:,42) == 0,:);
%anomalies
anakddinx = find(kddData(:,42) == 1);
rand1 = randperm(s, size(anakddinx,1));
%make proportion 1%
selectinx = anakddinx(rand1(1:1000));
newkddData = [normalkdd;kddData(selectinx,:)];
random=randperm(s, size(newkddData,1));
kddDatasmall = newkddData(random(1:20000),:);
data = kddDatasmall(:,1:41);
label = kddDatasmall(:,42);
%make traindata only normals
trainData = data(label == 0, :);
trainLabel(1:size(trainData,1),1) = 1; %1 normal 
testlabel(label == 0,1) = 1; %normal
testlabel(label ~= 0,1) = -1; %anomaly

%% SVM
%set nu range
m(1,:) = '-s 2 -t 2 -n 0.35';
m(2,:) = '-s 2 -t 2 -n 0.40';
m(3,:) = '-s 2 -t 2 -n 0.45';
m(4,:) = '-s 2 -t 2 -n 0.50';
m(5,:) = '-s 2 -t 2 -n 0.55';
m(6,:) = '-s 2 -t 2 -n 0.60';
m(7,:) = '-s 2 -t 2 -n 0.65';
m(8,:) = '-s 2 -t 2 -n 0.70';
m(9,:) = '-s 2 -t 2 -n 0.75';
m(10,:) = '-s 2 -t 2 -n 0.80';
n = 0.35:0.05:0.80;

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
xlim([0.35 0.8]);
xlabel('nu');
ylabel('value');
title('Kdd with OCSVM');
legend('precision', 'recall', 'fscore');