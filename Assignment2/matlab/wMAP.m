function [ output ] = wMAP( x, y, Phi, alpha, beta)
    PhiTrans = Phi';
    xSize = size(x);
    Sn_inv = alpha * eye(xSize(2)+1) + beta*(PhiTrans*Phi);
    Sn = inv(Sn_inv);
    Mn = beta*Sn*PhiTrans*y;
    output = Mn;
end

