clear all;
rng(42);
train = dlmread('sunspotsTrainStatML.dt');
test = dlmread('sunspotsTestStatML.dt');
l = length(train);

% Previous RMS values using wML:
rmsTest1 = 35.4651
rmsTest2 = 28.8398
rmsTest3 = 18.7700

Phi1 = horzcat(ones(l,1),train(:,3),train(:,4));
Phi2 = horzcat(ones(l,1),train(:,5));
Phi3 = horzcat(ones(l,1),train(:,1),train(:,2),train(:,3),train(:,4),train(:,5));

wML1 = wML(Phi1, train(:,6));
wML2 = wML(Phi2, train(:,6));
wML3 = wML(Phi3, train(:,6));

alphaValues = -100:1:-20;
rms1 = zeros(length(alphaValues),1);
rms2 = zeros(length(alphaValues),1);
rms3 = zeros(length(alphaValues),1);
smallerAlpha1 = [];
smallerAlpha2 = [];
smallerAlpha3 = [];

for i=1:length(alphaValues)
    alpha = alphaValues(i);
    wMAP1 = wMAP(train(:,3:4), train(:,6), Phi1, alpha, 1);
    wMAP2 = wMAP(train(:,5), train(:,6), Phi2, alpha, 1);
    wMAP3 = wMAP(train(:,1:5), train(:,6), Phi3, alpha, 1);
    pred_test1 = linearBasisFunction(test(:,3:4), wMAP1);
    pred_test2 = linearBasisFunction(test(:,5), wMAP2);
    pred_test3 = linearBasisFunction(test(:,1:5), wMAP3);
    rms1(i) = rootMeanSq( pred_test1, test(:, 6));
    rms2(i) = rootMeanSq( pred_test2, test(:, 6));
    rms3(i) = rootMeanSq( pred_test3, test(:, 6));

    %if rms1(i) < rmsTest1
    %    smallerAlpha1 = vertcat(smallerAlpha1,i);
    %end
    %if rms2(i) < rmsTest2
    %   smallerAlpha2 = vertcat(smallerAlpha2,i); 
    %end
    if rms3(i) < rmsTest3
       smallerAlpha3 = vertcat(smallerAlpha3,i); 
    end
end

figure;
hold on;
plot(alphaValues, rms1, alphaValues, rms2, alphaValues, rms3);
legend('Features 3/4', 'Feature 5', 'All features');
hold off;

smallestRms = min(rms3)