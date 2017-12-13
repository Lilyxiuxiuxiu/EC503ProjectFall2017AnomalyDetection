A = importdata('kddcup.data_10_percent_corrected');
for i = 1: size(A, 1)
    cellArray(i,:) = strsplit(char(A(i,:)), ',');
end
clear A

kddData = zeros(size(cellArray,1), size(cellArray,2));
kddData(:,1) = str2num(char(cellArray(:,1)));
for i = 5: 41
    i
    kddData(:,i) = str2num(char(cellArray(:,i)));
end

for i = 2:4
    [~, kddData(:,i)] = ismember(cellArray(:,i), unique(cellArray(:,i)));
end

[~, kddData(:,42)] = ismember(cellArray(:,42), unique(cellArray(:,42)));
inx = kddData(:,42) == 12;
kddData(inx == 1, 42) = 0;
kddData(inx == 0, 42) = 1;
clear cellArray