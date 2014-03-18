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
M = 2;

h = @(a) a/(1+abs(a));
dh = @(a) 1/(1+abs(a))^2;
learningRate = 0.01;
stopDifference = 0.03;


% Verification of gradients
[wMD, wKM] = initialWeightsNN(K,M,D);
update_wKM = zeros(K,M+1);
update_wMD = zeros(M+1,D+1);
for i=1:length(X)
    sample = X(:,i);
    [a,z,y_pred] = forwardProp(sample, h, wMD, wKM);
    y_true = y(i);
    [delta_wMD, delta_wKM, ~] = backwardProp( y_pred, y_true, dh, a, wKM, z, sample );
    update_wKM = update_wKM + delta_wKM;
    update_wMD = update_wMD + delta_wMD;
end

reference_wKM = (1/length(X)) .* update_wKM;
reference_wMD = (1/length(X)) .* update_wMD;
reference_wMD = reference_wMD(2:end,:)
reference_wKM

error = 0;
for i=1:length(X)
    sample = X(:,i);
    [~,~,y_pred] = forwardProp(sample, h, wMD, wKM);
    y_true = y(i);
    error = error + (y_pred - y_true)^2;
end
baseMSE = (1/(length(X))) * error

% wMD
epsilon = 10^(-8)
wMD_MSE = zeros(size(wMD,1), size(wMD,2));
for i=1:(size(wMD,1) * size(wMD,2))
    new_wMD = wMD;
    new_wMD(i) = new_wMD(i) + epsilon;
    error = 0;
    for j=1:length(X)
        sample = X(:,j);
        [~,~,y_pred] = forwardProp(sample, h, new_wMD, wKM);
        y_true = y(j);
        error = error + (y_pred - y_true)^2;
    end
    mse = ((1/(length(X))) * error);
    wMD_MSE(i) = ((mse - baseMSE) / epsilon) * 0.5;
end

% wKM
wKM_MSE = zeros(size(wKM,1), size(wKM,2));
for i=1:(size(wKM,1) * size(wKM,2))
    new_wKM = wKM;
    new_wKM(i) = new_wKM(i) + epsilon;
    error = 0;
    for j=1:length(X)
        sample = X(:,j);
        [~,~,y_pred] = forwardProp(sample, h, wMD, new_wKM);
        y_true = y(j);
        error = error + (y_pred - y_true)^2;
    end
    mse = ((1/(length(X))) * error);
    wKM_MSE(i) = ((mse - baseMSE) / epsilon) * 0.5;
end

diff_wMD = reference_wMD - wMD_MSE
diff_wKM = reference_wKM - wKM_MSE