function [ bestC, bestgamma ] = fivefoldSVM( trainX, trainY, testX, testY, Cs, gammas )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
sigmas = arrayfun(@(gamma) sqrt(1/(2*gamma)),gammas);
accs = zeros(length(Cs),length(gammas));

for i=1:length(Cs)
    C = Cs(i);
    for j=1:length(gammas)
        sigma = sigmas(j);
        % Create new train/test set.
        shuffled = shuffleSplit(trainX, trainY, 5);
        acc = zeros(1,5);
        for k=1:5
            partsToJoin = 1:5;
            partsToJoin(k) = [];
            [ltrainX, ltrainY] = bucketJoiner(shuffled, partsToJoin);
            [ltestX , ltestY ] = bucketJoiner(shuffled, (k));
            
            model = svmtrain(ltrainX, ltrainY, 'kernel_function','rbf','rbf_sigma',sigma,'boxconstraint',C);
            predY = svmclassify(model,ltestX);
            acc(k) = 1 - (nnz(predY - ltestY)) / length(predY);
        end
        accs(i,j) = mean(acc);
    end
end

[row,col] = find(accs == max(accs(:)));
bestC = Cs(row(1));
bestgamma = gammas(col(1));

end

