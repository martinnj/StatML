clear all;
rng(42); % Seed random generator to have consistent results.

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

meanX = mean(X);
meanY = mean(Y);
muML = [meanX; meanY];
plot(X, Y, 'x', meanX, meanY, 'o', mu(1), mu(2), 'o');
axis equal;
grid on;
title('Multivariate Gaussian distribution for N=100 with sample mean and distribution mean', 'FontSize', 15);
legend('Multivariate Gaussian Distribution', 'Sample mean', 'Distribution Mean');
