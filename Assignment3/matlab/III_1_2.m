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
sinc = @(x) sin(x) / x;
learningRate = 0.01;
stopDifference = 0.03;

% NN with M=2
[wMD, wKM, errors_train, errors_val] = nnTrain(X, y, X_validate, y_validate, h, dh, K, 2, D, learningRate, stopDifference);
epochs = length(errors_train);

% MSE on training set and validation set
figure;
semilogy(1:epochs, errors_val, 'xr', 1:epochs, errors_train, 'xb');
hold on, grid on;
title('Mean-Squared Error vs. Network epoch (2 hidden nodes)');
legend('Validation data MSE', 'Training data MSE');
xlabel('Network epoch');
ylabel('Mean-Squared Error');
hold off;

% NN with M=20
[wMD, wKM, errors_train, errors_val] = nnTrain(X, y, X_validate, y_validate, h, dh, K, 20, D, learningRate, stopDifference);
epochs = length(errors_train);

% MSE on training set and validation set
figure;
semilogy(1:epochs, errors_val, 'xr', 1:epochs, errors_train, 'xb');
hold on, grid on;
title('Mean-Squared Error vs. Network epoch (20 hidden nodes)');
legend('Validation data MSE', 'Training data MSE');
xlabel('Network epoch');
ylabel('Mean-Squared Error');
hold off;

rates = [0.001, 0.01, 0.1];
points = -10:0.05:10;
sincvalues = arrayfun(sinc, points);
preds = zeros(length(rates), length(points));
for i=1:length(rates)
    rate = rates(i);
    [wMD, wKM, errors_train, errors_val] = nnTrain(X, y, X_validate, y_validate, h, dh, K, 2, D, rate, stopDifference);
    for x=1:length(points)
       [~,~,preds(i,x)] = forwardProp([1; points(x)], h, wMD, wKM);
    end
end

% Plot networks vs. sinc(x) for 2 hidden nodes
figure;
plot(points, sincvalues, points, preds(1,:), points, preds(2,:), points, preds(3,:));
hold on;
title('Neural network with 2 hidden nodes with various learning rates');
legend('sinc(x)','NN(2 hidden) with learning rate 0.001','NN(2 hidden) with learning rate 0.01','NN(2 hidden) with learning rate 0.1');
xlabel('-10 to 10');
hold off;

preds = zeros(length(rates), length(points));
for i=1:length(rates)
    rate = rates(i);
    [wMD, wKM, errors_train, errors_val] = nnTrain(X, y, X_validate, y_validate, h, dh, K, 20, D, rate, stopDifference);
    for x=1:length(points)
       [~,~,preds(i,x)] = forwardProp([1; points(x)], h, wMD, wKM);
    end
end

% Plot networks vs. sinc(x) for 20 hidden nodes
figure;
plot(points, sincvalues,points, preds(1,:),points, preds(2,:),points, preds(3,:));
hold on;
title('Neural network with 20 hidden nodes with various learning rates');
legend('sinc(x)','NN(20 hidden) with learning rate 0.001','NN(20 hidden) with learning rate 0.01','NN(20 hidden) with learning rate 0.1');
xlabel('-10 to 10');
hold off;