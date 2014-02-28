clear all;

train = dlmread('IrisTrain2014.dt');
test = dlmread('IrisTest2014.dt');

X_train = train(:,1:2);
y_train = train(:,3);

X_test = test(:,1:2);
y_test = test(:,3);

X_test = scale(X_train, X_test);
X_train = scale(X_train, X_train);

% Priors = Percentage of dataset belonging to the given class.
class0_idx = find(y_train == 0);
class1_idx = find(y_train == 1);
class2_idx = find(y_train == 2);

l_k = [length(class0_idx); 
       length(class1_idx); 
       length(class2_idx)];

priors = l_k ./ length(y_train)

X_class0 = X_train(class0_idx,:);
X_class1 = X_train(class1_idx,:);
X_class2 = X_train(class2_idx,:);  

mu0 = 1 / l_k(1) * sum(X_class0);
mu1 = 1 / l_k(2) * sum(X_class1);
mu2 = 1 / l_k(3) * sum(X_class2);

innerCov0 = zeros(2,2);
for i=1:length(class0_idx) - 3
    idx = class0_idx(i);
    v = X_train(idx,:) - mu0;
    % Even though formula is v*v', we need v'*v because
    % our data is turned the wrong way according to what
    % the formula seems to expect.
    innerCov0 = innerCov0 + (v' * v);
end

innerCov1 = zeros(2,2);
for i=1:length(class1_idx) - 3
    idx = class0_idx(i);
    v = X_train(idx,:) - mu1;
    % Even though formula is v*v', we need v'*v because
    % our data is turned the wrong way according to what
    % the formula seems to expect.
    innerCov1 = innerCov1 + (v' * v);
end

innerCov2 = zeros(2,2);
for i=1:length(class2_idx) - 3
    idx = class2_idx(i);
    v = X_train(idx,:) - mu2;
    % Even though formula is v*v', we need v'*v because
    % our data is turned the wrong way according to what
    % the formula seems to expect.
    innerCov2 = innerCov2 + (v' * v);
end

outerCov = innerCov0 + innerCov1 + innerCov2;
averageEstCov = 1/(length(X_train) - 3) * outerCov

pred_train = zeros(length(X_train),1);
for i=1:length(X_train)
   x = X_train(i,:);
   x_class = y_train(i);
   d0 = linearDecision(x, averageEstCov, mu0, priors(1));
   d1 = linearDecision(x, averageEstCov, mu1, priors(2));
   d2 = linearDecision(x, averageEstCov, mu2, priors(3));
   d = [d0; d1; d2];
   [~,I] = max(d);
   pred_train(i) = I-1;
end

pred_test = zeros(length(X_test),1);
for i=1:length(X_test)
   x = X_test(i,:);
   x_class = y_test(i);
   d0 = linearDecision(x, averageEstCov, mu0, priors(1));
   d1 = linearDecision(x, averageEstCov, mu1, priors(2));
   d2 = linearDecision(x, averageEstCov, mu2, priors(3));
   d = [d0; d1; d2];
   [~,I] = max(d);
   pred_test(i) = I-1;
end

normAccTrain    = 1 - (nnz(pred_train - y_train) / length(y_train))
normAccTest     = 1 - (nnz(pred_test - y_test) / length(y_test))