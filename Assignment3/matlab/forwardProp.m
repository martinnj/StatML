function [ out, a ] = forwardProp( x, wMD, wKM, y_act, h )
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
    K = size(wKM,1);
    M = size(wMD,1);
    D = size(wMD,2);
    % Step 1: Calculate vector of hidden node values.
    a = wMD * x;
    % Step 2: Calculate vector of output node values, applying
    % the activation function h() to each entry. See Eq 5.7, 5.8, 5.9 in
    % book.
    z = arrayfun(h, a);
    out = wKM * z;
end