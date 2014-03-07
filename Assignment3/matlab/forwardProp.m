function [ out ] = forwardProp( X, wMD, wKM, y_act )
% Invariants: 
%   K=Number of output neurons
%   D=Dimensionality of input samples
%   M=Number of hidden neurons
%
%   X is a matrix where each sample is a column
%   and the first row must be all 1's.
%
%   wMD is an M x D matrix
%   wKM is a  K x M matrix
%  
%   The first row of wMD should be [1; 0; 0; ...; 0]
    h = @(a) a/(1+abs(a));
    D = length(X);
    wMDSize = size(wMD);
    out = y_act(wKM * wMD * X)';
end