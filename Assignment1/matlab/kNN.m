function [ Results ] = kNN( K, X, y, distFunc, testX)
% For each input value x in testX:
%   1) Creates a Distances table where each row is of the form
%       [distance class]
%   2) Sorts the Distances table by the first element (distance to x)
%   3) Collects the K nearest neighbors.
%   4) Returns the mode of the nearest neighbors(most frequent neighbor).
    Results = zeros(length(testX),1);

    if K > length(X)
       error('K must be smaller than the number of training points!') 
    end
    
    if K < 1
        error('K must be greater than 0.')
    end

    for i= 1:length(testX)
        Distances = zeros(length(X), 2);
        for j= 1:length(X)
            Distances(j, 1) = distFunc(X(j,:), testX(i,:));
            Distances(j, 2) = y(j);
        end
        Distances = sortrows(Distances);
        KClosest = Distances(1:K, 2);
        Results(i,1) = mode(KClosest);
    end
end