clear all;
rng(42);

train       = dlmread('sincTrain25.dt');
validate    = dlmread('sincValidate10.dt');

X = vertcat(ones(1,length(train)), train(:,1)');
y = train(:,2);

K = 1;
D = 1;
M = 2;


h = @(a) a/(1+abs(a));
dh = @(a) 1/(1+abs(a))^2;
id = @(x) x;
mse = @(errors) mean(errors.^2);

[wMD,wKM] = initialWeightsNN(K, M, D);

learningRate = 0.01;
stopDifference = 0.1;
diffKM = 1;
diffMD = 1;
counter = 0;

while diffKM > 0 || diffMD > 0
    counter = counter + 1;
    update_wKM = zeros(K,M+1);
    update_wMD = zeros(M+1,D+1);
    for x=1:length(X)
        % Forward propagation
        sample = X(:,x);
        a = wMD * sample;
        z = arrayfun(h, a);
        z(1) = 1;
        y_pred = wKM * z;

        % Backpropagation
        % Get output delta's
        % Do we square this or not??
        y_delta = y_pred - y(x);

        % Apply h'() to each a_j
        a_d = arrayfun(dh, a);
        a_d(1) = 1;

        % Get hidden delta's using (5.66) i.e. h'(a) * w_kj * delta_y
        hidden_delta = a_d .* (wKM*y_delta)';

        % Update wKM (See 5.67, 5.27)
        update_wKM = update_wKM + (y_delta * z');
        % Update wMD (See 5.67, 5.27)
        update_wMD = update_wMD + (hidden_delta * sample');
    end
    
    % Update
    wKM_new = wKM - learningRate * update_wKM;
    wMD_new = wMD - learningRate * update_wMD;
    
    diffKM = nnz(arrayfun(@abs, update_wKM) > stopDifference);
    diffMD = nnz(arrayfun(@abs, update_wMD) > stopDifference);
    
    wKM = wKM_new;
    wMD = wMD_new;
end
% Step 2: Calculate the output node error.
%error = output - y

% Step 3: Calculate hidden node errors.
%hidden_errors = zeros(1,M+1);
%for j=1:M+1
%    v = dh(output(j));
    % OMG WORST CODE EVER> USE A SUM, F00
%    s = 0;
%    for k=1:K
%        s = s + (wKM(k,j) * output(k));
%    end
%    hidden_errors(j) = v*s;
%end

%wKM = wKM - hidden_errors
%wMD = wMD - hidden_errors
% TODO: Do something with the hidden errors!