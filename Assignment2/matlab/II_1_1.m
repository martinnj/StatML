% Begun testing according to:
% http://www.mathworks.se/help/stats/discriminant-analysis.html

clear all;

train = dlmread('IrisTrain2014.dt');
test = dlmread('IrisTest2014.dt');

L = train(:,1);
W = train(:,2);
C = train(:,3);

%gscatter(L,W,C);

X = [L,W];
class = ClassificationDiscriminant.fit(X,C);