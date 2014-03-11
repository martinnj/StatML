clear all;
rng(42);

train     = dlmread('parkinsonsTrainStatML.dt');
test      = dlmread('parkinsonsTestStatML.dt');

% X = Input vectors
trainSize = size(train);
trainSize = trainSize(2);
testSize  = size(test);
testSize  = testSize(2);

trainX = train(:, 1:trainSize-1);
testX  = test(:, 1:testSize-1);
priorTrainMean = mean(trainX);
priorTrainVar  = var(trainX);

trainY = train(:, trainSize);
testY = test(:, testSize);

% Get stuff before it's normalized.
Cs = [0.01, 0.1, 1, 10, 100, 1000, 10000];
gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100];
[preNormBestC, preNormBestgamma, preNormBestAcc] = fivefoldSVM(trainX, trainY, testX, testY, Cs, gammas)

% Normalize it!
testX = scale(trainX,testX);
trainX = scale(trainX,trainX);
trainMean = mean(trainX);
trainVar = var(trainX);
testMean = mean(testX);
testVar = var(testX);

trainY = train(:, trainSize);
testY = test(:, testSize);


% Lets test after data normalization.
[normBestC, normBestgamma, normBestAcc] = fivefoldSVM(trainX, trainY, testX, testY, Cs, gammas)