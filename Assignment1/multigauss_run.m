mu = [1,2]';
Sigma = [0.3,0.2;0.2,0.2];
L = chol(Sigma,'lower');
N=100;
points=zeros(2,N);
for j=1:N,
    z = randn(2,1);
    points(:, j) = multigauss( mu, L, z );
    %points(:, multigauss( mu, L, z ))
end

X = points(1,:);
Y = points(2,:);
meanX = mean(X);
meanY = mean(Y);
plot(X,Y,'x',meanX,meanY,'o',mu(1),mu(2),'o');
title('Multivariate Gaussian Distribution for N=100','FontSize',15);
grid on;
legend('Multivariate Gaussian Distribution','Mean X/Y','Mu');
[meanX,meanY]
mu'
norm([meanX,meanY] - mu')