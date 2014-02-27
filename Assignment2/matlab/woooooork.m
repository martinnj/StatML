clear all;
rng(42);
train = dlmread('sunspotsTrainStatML.dt');

l = length(train);

Phi1 = horzcat(ones(l,1),train(:,3),train(:,4));
Phi2 = horzcat(ones(l,1),train(:,5));
Phi3 = horzcat(ones(l,1),train(:,1),train(:,2),train(:,3),train(:,4),train(:,5));

wML1 = wML(Phi1, train(:,6))
wML2 = wML(Phi2, train(:,6))
wML3 = wML(Phi3, train(:,6))