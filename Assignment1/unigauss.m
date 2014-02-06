function y = unigauss( x, mu, sigma )
normalize = 1/(sigma * sqrt(2*pi));
exponent = -((x-mu).^2/(2*sigma.^2));

y = normalize*exp(1).^exponent;

end

