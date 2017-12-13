A = importdata('pen-global-unsupervised-ad.csv');
for i = 1: size(A, 1)
    cellArray(i,:) = strsplit(char(A(i,:)), ',');
end


pg_data =  zeros(size(cellArray,1), size(cellArray,2));
for i = 1: 16
    i
    pg_data(:,i) = str2num(char(cellArray(:,i)));
end

[~, pg_data(:,17)] = ismember(cellArray(:,17), unique(cellArray(:,17)));