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
trainY = train(:, trainSize);
testY = test(:, testSize);

sigma = sqrt(1/(2*0.1));
C = 10;

model = svmtrain(trainX, trainY, 'kernel_function','rbf','rbf_sigma',sigma,'boxconstraint',C);

% ~ returns 1 if element is 0; 0 otherwize
numberOfBoundedX = nnz(~arrayfun(@(x) abs(x)-C, model.Alpha));

Cs = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 10000];
lengthCs = length(Cs);
numberOfBoundedXs = zeros(1,lengthCs);
numberOfFreeXs = zeros(1,lengthCs);
%alphas = zeros(lengthCs, 100);

for i = 1:lengthCs
    model = svmtrain(trainX, trainY, 'kernel_function','rbf','rbf_sigma',sigma,'boxconstraint',Cs(i));
    %alphas(i,:) = vertcat(model.Alpha./Cs(i), zeros(100-length(model.Alpha), 1));
    number = nnz(arrayfun(@(x) abs(x./Cs(i)) <= 1, model.Alpha));
    numberOfBoundedXs(1,i) = number;
    numberOfFreeXs(1,i) = length(model.Alpha) - number;
end

