clear all;
rng(42);

train       = dlmread('sincTrain25.dt');
validate    = dlmread('sincValidate10.dt');

X = vertcat(ones(1,length(train)), train(:,1)');
y = train(:,2);
%X_test = vertcat(ones(1,length(train)), validate(:,1)');
%y_test = validate(:,2);

K = 1;
D = 1;
M = 20;

h = @(a) a/(1+abs(a));
dh = @(a) 1/(1+abs(a))^2;
learningRate = 0.01;
stopDifference = 0.03;

[wMD, wKM, errors] = nnTrain(X, y, h, dh, K, M, D, learningRate, stopDifference)