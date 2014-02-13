%%%% I.2.2
mu = [1,2]';
Sigma = [0.3,0.2;0.2,0.2];
L = chol(Sigma,'lower');
N=100;
points=zeros(2,N);
for j=1:N,
    z = randn(2,1);
    points(:, j) = multigauss( mu, L, z );
end

X = points(1,:);
Y = points(2,:);

plot(X, Y, 'x');
title('Multivariate Gaussian distribution for N=100', 'FontSize', 15);

%%%% I.2.3
meanX = mean(X);
meanY = mean(Y);
plot(X, Y, 'x', meanX, meanY, 'o', mu(1), mu(2), 'o');
title('Multivariate Gaussian distribution for N=100 with sample mean and distribution mean', 'FontSize', 15);
legend('Multivariate Gaussian Distribution', 'Sample mean', 'Distribution Mean');


%%%%% I.2.4
%%%%% SigmaML = Maximum likelihood
sigmaML = zeros(2,2);
for i=1:N,
    sigmaML = sigmaML + (points(:,i) - mu) * (points(:,i) - mu)';
end
sigmaML = sigmaML ./ N;

[eigenVectors,eigenValues] = eig(sigmaML);

%%%%% e1 and e2 = Scaled and translated eigenvectors, see assignment.
e1 = mu + sqrt(eigenValues(1))*eigenVectors(:,1);
e2 = mu + sqrt(eigenValues(2))*eigenVectors(:,2);

figure;
tx = [mu(1); e1(1)];
ty = [mu(2); e1(2)];
plot(tx, ty);
hold on;
tx_1 = [mu(1); e2(1)];
ty_1 = [mu(2); e2(2)];
plot(tx_1, ty_1);
plot(X,Y,'o');
hold off;
title('TEST');
%plot(X, Y, 'x', meanX, meanY, 'o', mu(1), mu(2), 'o', );
%title('Multivariate Gaussian distribution for N=100 with sample mean and distribution mean', 'FontSize', 15);
%legend('Multivariate Gaussian Distribution', 'Sample mean', 'Distribution Mean');