A = importdata('iris.data');
for i = 1: size(A, 1)
    cellArray(i,:) = strsplit(char(A(i,:)), ',');
end

iris_data =  zeros(size(cellArray,1), size(cellArray,2));
for i = 1: 4
    i
    iris_data(:,i) = str2num(char(cellArray(:,i)));
end

[~, iris_data(:,5)] = ismember(cellArray(:,5), unique(cellArray(:,5)));

iris_sanota_data = iris_data;
iris_sanota_data(iris_sanota_data(:,5)~= 1,5) = 2; %anomaly

iris_virigi_data = iris_data;
iris_virigi_data(iris_virigi_data(:,5) ~= 2,5) = 1; %anomaly

iris_versic_data = iris_data;
iris_versic_data(iris_versic_data(:,5) ~= 3,5) = 2; %abnormal
iris_versic_data(iris_versic_data(:,5) == 3,5) = 1; %normal
