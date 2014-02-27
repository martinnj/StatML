function [ out ] = wML( Phi, t)
    out = pinv(Phi'*Phi) * (Phi'*t);
end

