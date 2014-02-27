function [ out ] = wML( Phi, t, D )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
tD = [];
for i=1:D
    tD = vertcat(tD,t);
end

out = pinv(Phi'*Phi) * (Phi'*tD);

end

