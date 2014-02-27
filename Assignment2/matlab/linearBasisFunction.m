function [ pred_y ] = linearBasisFunction( x, w )
    dimensions = size(x);
    M = x';
    M = vertcat(ones(1,dimensions(1)), M);
    pred_y = w'*M;
    pred_y = pred_y';
end

