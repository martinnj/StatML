clear all;
rng(42);

train       = dlmread('sincTrain25.dt');
validate    = dlmread('sincValidate10.dt');

X = vertcat(ones(1,length(train)), train(:,1)')
y = train(:,2)

K = 1;
D = 1;
M = 2;

[wMD,wKM] = initialWeightsNN(K, M, D);

firstTry = forwardProp(X, wMD, wKM, @(x) x);
error = firstTry - y