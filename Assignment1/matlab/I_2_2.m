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

figure;
plot(X, Y, 'x');
title('Multivariate Gaussian distribution for N=100', 'FontSize', 15);