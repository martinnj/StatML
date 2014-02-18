clear all;

train = dlmread('IrisTrain2014.dt');
test = dlmread('IrisTest2014.dt');
KValues = [1 3 5];

% X = Input vectors
trainSize = size(train);
trainSize = trainSize(2);
testSize = size(test);
testSize = testSize(2);

trainX = train(:, 1:trainSize-1);
testX = test(:, 1:testSize-1);

% y = Target classes
trainY = train(:, 3);
testY = test(:, 3);

% Results is a list of pairs (KValue, 0-1 loss)
Results = zeros(2*length(KValues), 2);

for i=1:length(KValues)
   K = KValues(i);
   Results(2*i-1,1) = K;
   Results(2*i,1) = K;
   TrainPred = kNN(K, trainX, trainY, @eucl, trainX);
   TestPred = kNN(K, trainX, trainY, @eucl, testX);
   Results(2*i-1,2) = 1 - (nnz(trainY - TrainPred)) / length(trainY);
   Results(2*i,2) = 1 - (nnz(testY - TestPred)) / length(testY);
end
Results