clear all;
rng(42);

train       = dlmread('sincTrain25.dt');
validate    = dlmread('sincValidate10.dt');

X = vertcat(ones(1,length(train)), train(:,1)');
y = train(:,2);

K = 1;
D = 1;
M = 2;

h = @(a) a/(1+abs(a));
dh = @(a) 1/(1+abs(a))^2;
id = @(x) x;

[wMD,wKM] = initialWeightsNN(K, M, D);

% Step 1: Forward Propagation.
output = forwardProp(X, wMD, wKM, id, h);

% Step 2: Calculate the output node errors.
error = output - y

% Step 3: Calculate hidden node errors.
hidden_errors = zeros(1,M+1);
for j=1:M+1
    v = dh(output(j));
    % OMG WORST CODE EVER> USE A SUM, F00
    s = 0;
    for k=1:K
        s = s + (wKM(k,j) * output(k));
    end
    hidden_errors(j) = v*s;
end
hidden_errors
% TODO: Do something with the hidden errors!