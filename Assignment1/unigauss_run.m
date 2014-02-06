% First gauss run:
x = -10:0.01:10;
y1 = gaus(x,-1,1);

% Second run
y2 = gaus(x,0,2);

% Third run
y3 = gaus(x,2,3);
plot(x,y1,x,y2,x,y3);
title('Gaussian distributions','FontSize',15);
grid on;
legend('(-1,1)','(0,2)','(2,3)')
