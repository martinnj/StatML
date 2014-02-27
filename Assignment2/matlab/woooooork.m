clear all;
rng(42);
train = dlmread('sunspotsTrainStatML.dt');
test = dlmread('sunspotsTestStatML.dt');
l = length(train);

Phi1 = horzcat(ones(l,1),train(:,3),train(:,4));
Phi2 = horzcat(ones(l,1),train(:,5));
Phi3 = horzcat(ones(l,1),train(:,1),train(:,2),train(:,3),train(:,4),train(:,5));

wML1 = wML(Phi1, train(:,6));
wML2 = wML(Phi2, train(:,6));
wML3 = wML(Phi3, train(:,6));

pred_test1 = linearBasisFunction(test(:,3:4), wML1);
pred_test2 = linearBasisFunction(test(:,5), wML2);
pred_test3 = linearBasisFunction(test(:,1:5), wML3);

figure;
hold on;
title('x and t of training set plotted with real and predicted target variables of test set.');
plot(train(:,5),train(:,6),'bx', test(:,5), test(:,6), 'rx', test(:,5), pred_test2, 'gx');
hold off;
rmsTest1 = rootMeanSq(pred_test1, test(:,6))
rmsTest2 = rootMeanSq(pred_test2, test(:,6))
rmsTest3 = rootMeanSq(pred_test3, test(:,6))

years = 1916:2011;

figure;
hold on;
title('Actual(blue) vs. predicted(red) sunspots from 1916 - 2011 using selection 1');
plot(years, pred_test1, 'r', years, test(:,6), 'b');
hold off;
figure;
hold on;
title('Actual(blue) vs. predicted(red) sunspots from 1916 - 2011 using selection 2');
plot(years, pred_test2, 'r', years, test(:,6), 'b');
hold off;
figure;
hold on;
title('Actual(blue) vs. predicted(red) sunspots from 1916 - 2011 using selection 3');
plot(years, pred_test3, 'r', years, test(:,6), 'b');
hold off;