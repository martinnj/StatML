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

% Compute gradients using arbitrary sample data.
[wMD, wKM, errors] = nnTrain(X, y, h, dh, K, M, D, learningRate, stopDifference);


hold on;
plot(X(2,:),y,'ob');
 
preds = zeros(1,length(X));
for x=1:length(X)
    [~,~,y_pred] = forwardProp(X(:,x), h, wMD, wKM);
    preds(1,x) =  y_pred;
end
plot(X(2,:),preds(1,:),'or');