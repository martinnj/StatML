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
priorTrainMean = mean(trainX)
priorTrainVar  = var(trainX)

% Normalize it!
testX = scale(trainX,testX);
trainX = scale(trainX,trainX);
trainMean = mean(trainX)
trainVar = var(trainX)
testMean = mean(testX)
testVar = var(testX)