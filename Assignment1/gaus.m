function y = gaus( x, nu, sigma )
normalize = 1/(sigma * sqrt(2*pi));
exponent = -((x-nu).^2/(2*sigma.^2));

y = normalize*exp(1).^exponent;

end

