% 2017 EC503 Project pen-global gknn
%% load data
load('pg_data.mat')
s = RandStream('mt19937ar','Seed',0);
rand = randperm(s, 809);
label = pg_data(rand,17);
data = pg_data(rand,1:16);

%% get k neighbors
%calculate distance between  points
distance = squareform(pdist(data));
%sort distance row-wise
sorted = sort(distance,2);
count = 1;

% TP FP FN TN
onerate = zeros(8,4);
tworate = zeros(8,4);
for k = 1:25:200
    k_nn_dis = sorted(:,2:k+1);
    avg_dis = mean(k_nn_dis,2);
    
    figure(1)
    subplot(2, 4, count);
    ana_dis = avg_dis(label == 2);
    nor_dis = avg_dis(label == 1);
    A = 1:size(data,1);
    scatter(A(label == 1), nor_dis, 'b');
    hold on
    scatter(A(label == 2), ana_dis,  'r');
    hold off
    legend('normal data points', 'anomaly data points');
    xlabel('n');
    ylabel('average distance');
    xlim([0,850]);
    ylim([0,180]);
    title(k);
    
    figure(2)
    subplot(2, 4, count);
    histogram(avg_dis, 100)
    xlabel('average distance');
    ylabel('count of each bin');
    title(k);
    
    
    %threshold 1std
    onestd = mean(avg_dis) + 1* std(avg_dis);
    %set the distance larger than threshold
    one_anormly_inx = avg_dis > onestd;
    one_prediction = ones(809,1);
    one_prediction(one_anormly_inx) = 2;
    fprintf('%d-nn', k);
    oneconf = confusionmat(one_prediction, label);
    printmat(oneconf, '1 std Confusion Matrix', 'PredNormal PredAnomaly', 'GTNormal GTAnomaly');
    
    
    onerate(count,1) = oneconf(2,2);
    onerate(count,2) = oneconf(2,1);
    onerate(count,3) = oneconf(1,2);
    onerate(count,4) = oneconf(1,1);
    
    
    %threshold 2std
    twostd = mean(avg_dis) + 2* std(avg_dis);
    %set the distance larger than threshold
    two_anormly_inx = avg_dis > twostd;
    two_prediction = ones(809,1);
    two_prediction(two_anormly_inx) = 2;
    twoconf = confusionmat(two_prediction, label);
    printmat(twoconf, '2 std Confusion Matrix', 'PredNormal PredAnomaly', 'GTNormal GTAnomaly');
    
    
    tworate(count,1) = twoconf(2,2);
    tworate(count,2) = twoconf(2,1);
    tworate(count,3) = twoconf(1,2);
    tworate(count,4) = twoconf(1,1);
    
    
    count = count +1;
    
end

%% analysis
oneprecision = onerate(:,1)./(onerate(:,1) + onerate(:,2));
onerecall = onerate(:,1)./(onerate(:,1) + onerate(:,3));
onefscore = 2 * oneprecision.*onerecall ./(oneprecision +onerecall);

twoprecision = tworate(:,1)./(tworate(:,1) + tworate(:,2));
tworecall = tworate(:,1)./(tworate(:,1) + tworate(:,3));
twofscore = 2 * twoprecision.*tworecall ./(twoprecision +tworecall);

figure(3)
k = 1:25:200;
plot(k,oneprecision, '--');
hold on
plot(k,twoprecision, '--');
plot(k,onerecall, ':','LineWidth',2);
plot(k,tworecall, ':','LineWidth',2);
plot(k,onefscore);
plot(k,twofscore);
hold off
xlabel('number of neighbors');
ylabel('value');
title('plot of 1std 2std precision, recall and fscore');
legend('1std precision', '2std precision', '1std recall', '2std recall', '1std fscore', '2std fscore');