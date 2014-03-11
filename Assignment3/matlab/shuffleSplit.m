function [ buckets ] = shuffleSplit( inputX, inputY, noOfBuckets )
  rng(42);
  dataLen = length(inputX);
  indices = randperm(dataLen);
  for i=1:noOfBuckets
    begin = floor((i-1)*(dataLen)/noOfBuckets+1);
    ending = floor((i)*(dataLen)/noOfBuckets);
    buckets{i,1} = inputX(indices(begin:ending),:);
    buckets{i,2} = inputY(indices(begin:ending),:);
  end
end

