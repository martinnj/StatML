clear all;
rng(42); % Seed random generator to have consistent results.

% First gauss run:
x = -10:0.01:10;
y1 = unigauss(x,-1,1);

% Second run
y2 = unigauss(x,0,2);

% Third run
y3 = unigauss(x,2,3);

figure;
hold on;
plot(x,y1,x,y2,x,y3);
title('Gaussian distributions','FontSize',15);
grid on;
hold off;
legend('(-1,1)','(0,2)','(2,3)')