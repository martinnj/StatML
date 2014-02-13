clear all;
rng(42); % Seed random generator to hvae consistent results.

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
muML = [meanX; meanY];
plot(X, Y, 'x', meanX, meanY, 'o', mu(1), mu(2), 'o');
axis equal;
grid on;
title('Multivariate Gaussian distribution for N=100 with sample mean and distribution mean', 'FontSize', 15);
legend('Multivariate Gaussian Distribution', 'Sample mean', 'Distribution Mean');

%%%%% I.2.4
%%%%% SigmaML = Maximum likelihood
SigmaML = zeros(2,2);
for i=1:N,
    SigmaML = SigmaML + (points(:,i) - muML) * (points(:,i) - muML)';
end
SigmaML = SigmaML ./ N

[eigenVectors,eigenValues] = eig(SigmaML);

%%%%% e1 and e2 = Scaled and translated eigenvectors, see assignment.
e1 = mu + sqrt(eigenValues(1,1))*eigenVectors(:,1);
e2 = mu + sqrt(eigenValues(2,2))*eigenVectors(:,2);

% Figure showing eigenvectors plotted onto the gaussian distribution.
figure;
ev1x = [mu(1); e1(1)];
ev1y = [mu(2); e1(2)];
ev2x = [mu(1); e2(1)];
ev2y = [mu(2); e2(2)];
hold on;
plot(X, Y, 'x', mu(1), mu(2), 'o');
plot(ev1x, ev1y,'-', 'Color', 'red');
plot(ev2x, ev2y,'-', 'Color', 'green');
hold off;
axis equal;
grid on;
title('Eigenvectors plotted onto Gaussian distribution, centered at mu.','FontSize',15);
legend('Multivariate Gaussian distribution', 'Distribution mean', 'Eigen vector 1', 'Eigen vector 2');

%%%% Rotating the covariance ^_^
figure;
hold on;
Sigma30 = rotateCov(SigmaML, 30);
Sigma60 = rotateCov(SigmaML, 60);
Sigma90 = rotateCov(SigmaML, 90);

%%%% Copy pasted plotting code
%%%% TODO: Replace with a function. Would be prettier.
L = chol(Sigma30,'lower');
points=zeros(2,N);
for j=1:N,
    z = randn(2,1);
    points(:, j) = multigauss( mu, L, z );
end
X = points(1,:);
Y = points(2,:);
plot(X, Y, 'x','Color','red');

L = chol(Sigma60,'lower');
points=zeros(2,N);
for j=1:N,
    z = randn(2,1);
    points(:, j) = multigauss( mu, L, z );
end
X = points(1,:);
Y = points(2,:);
plot(X, Y, 'x','Color','blue');

L = chol(Sigma90,'lower');
points=zeros(2,N);
for j=1:N,
    z = randn(2,1);
    points(:, j) = multigauss( mu, L, z );
end
X = points(1,:);
Y = points(2,:);
plot(X, Y, 'x','Color','magenta');

%%%% More stuff
[eigenVectors2,eigenValue2] = eig(SigmaML);
v = eigenVectors2(:,1);
angleOfX = -atand(v(1)/v(2));
SigmaX = rotateCov(SigmaML, angleOfX);

L = chol(SigmaX,'lower');
points=zeros(2,N);
for j=1:N,
    z = randn(2,1);
    points(:, j) = multigauss( mu, L, z );
end
X = points(1,:);
Y = points(2,:);
plot(X, Y, 'x','Color','black');
axis equal;
grid on;
title('Rotated Multivariate Gaussian distribution for N=100', 'FontSize', 15);
legend('Sigma rotated 30 degrees','Sigma rotated 60 degrees','Sigma rotated 90 degrees','Sigma rotated to match X-axis')
hold off;