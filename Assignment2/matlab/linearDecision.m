function [ score ] = linearDecision( x, Sigma, mu, prior)
    invSigma = inv(Sigma);
    score = x * invSigma * mu' - 1/2 * mu * invSigma * mu' + log(prior)
end

