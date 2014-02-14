function [ Xx ] = scale( Xbase, Xin )
X  = bsxfun(@minus, Xin, mean(Xbase));
Xx = X/std(X(:));
%Xx = bsxfun(@eucl, X(:,1), X(:,2));
end

