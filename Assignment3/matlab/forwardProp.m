function [ out ] = forwardProp( X, wMD, wKM, y_act, h )
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
    out = zeros(K,length(X));
    
    % For every sample
    for x=1:length(X)
        sample = X(:,x);
        % For every output node
        for k=1:K
            z = zeros(1,M);
            % For every hidden node
            for j=1:M    
                % For every input node
                for i=1:D
                    w_ji = wMD(j,i);
                    z(1,j) = z(1,j) + (w_ji * sample(i)); % Update value in hidden node j
                end
                z(1,j) = h(z(1,j)); % Apply activation function to the hidden node j.
            end
            % Output node k for sample x
            out(k,x) = y_act(sum(z));
        end
    end
    out = out';
end