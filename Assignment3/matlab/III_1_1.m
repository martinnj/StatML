clear all;
rng(42);

train       = dlmread('sincTrain25.dt');
validate    = dlmread('sincValidate10.dt');

X = vertcat(ones(1,length(train)), train(:,1)');
y = train(:,2);
X_validate = vertcat(ones(1,length(validate)), validate(:,1)');
y_validate = validate(:,2);

K = 1;
D = 1;
M = 20;

h = @(a) a/(1+abs(a));
dh = @(a) 1/(1+abs(a))^2;
learningRate = 0.01;
stopDifference = 0.03;

% Compute gradients using arbitrary sample data.
[wMD, wKM, errors_train, errors_val] = nnTrain(X, y, X_validate, y_validate, h, dh, K, M, D, learningRate, stopDifference);