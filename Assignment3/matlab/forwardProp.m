function [ a, z, y_pred ] = forwardProp( x, h, wMD, wKM )
    a = wMD * x;
    a = vertcat([1], a);
    z = arrayfun(h, a);
    y_pred = wKM * z;
end