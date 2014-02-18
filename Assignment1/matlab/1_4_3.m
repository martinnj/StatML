clear all;
rng(42); % Seed random generator to have consistent results.

train = dlmread('IrisTrain2014.dt');
test = dlmread('IrisTest2014.dt');
KValues = 1:2:25;

% X = Input vectors
trainSize = size(train);
trainSize = trainSize(2);
testSize = size(test);
testSize = testSize(2);

trainX = train(:, 1:trainSize-1);
testX = test(:, 1:testSize-1);
trainMean = mean(trainX(:))
trainVar = var(trainX(:))

% Normalize it!
testX = scale(trainX,testX);
trainX = scale(trainX,trainX);
testMean = mean(testX(:))
testVar = var(testX(:))

% y = Target classes
trainY = train(:, 3);
testY = test(:, 3);

shuffled = shuffleSplit(trainX, trainY, 5);

AvgLoss = zeros(length(KValues),2);
for i=1:length(KValues)
  k = KValues(i);
  Loss = zeros(5,1);
  for j=1:5
    [LtestX,LtestY] = bucketJoiner(shuffled, (j));
    partsToJoin = 1:5;
    partsToJoin(j) = [];
    [LtrainX, LtrainY] = bucketJoiner(shuffled, partsToJoin);
    PredY = kNN(k, LtrainX, LtrainY, @eucl, LtestX);
    Loss(j,1) = 1 - (nnz(PredY - LtestY)) / length(PredY);
  end
  AvgLoss(i,1) = k;
  AvgLoss(i,2) = mean(Loss(:,1));
end

sortedAvg = sortrows(AvgLoss,2);
normalizedBestK = sortedAvg(length(sortedAvg))

PredTest = kNN(normalizedBestK, trainX, trainY, @eucl, testX);
NormalizedPerformanceTest = 1 - (nnz(PredTest - testY)) / length(PredTest)

PredTrain = kNN(normalizedBestK, trainX, trainY, @eucl, trainX);
NormalizedPerformanceTrain = 1 - (nnz(PredTrain - trainY)) / length(PredTrain)