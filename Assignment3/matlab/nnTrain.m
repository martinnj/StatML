function [ wMD, wKM, errors, errors_val ] = nnTrain(X, y, X_validate, y_validate, h, dh, K, M, D, learningRate, stopDifference )

    [wMD,wKM] = initialWeightsNN(K, M, D);
    mse = @(errors) mean(errors.^2);
    diffKM = 1;
    diffMD = 1;
    counter = 0;
    errors = [];
    errors_val = [];

    while diffKM > 0 || diffMD > 0
        counter = counter + 1;
        update_wKM = zeros(K,M+1);
        update_wMD = zeros(M+1,D+1);
        round_errors = zeros(1,length(X));

        for x=1:length(X)
            % Forward propagation
            sample = X(:,x);
            [a, z, y_pred] = forwardProp(sample, h, wMD, wKM);

            % Backpropagation
            % Get output delta's
            y_delta = y_pred - y(x);
            round_errors(1,x) = y_delta^2;

            % Apply h'() to each a_j
            a_d = arrayfun(dh, a);

            % Get hidden delta's using (5.66) i.e. h'(a) * w_kj * delta_y
            hidden_delta = a_d .* (wKM*y_delta)';

            % Update wKM (See 5.67, 5.27)
            update_wKM = update_wKM + (y_delta * z');
            % Update wMD (See 5.67, 5.27)
            update_wMD = update_wMD + (hidden_delta * sample');
        end

        % Calculate RMS of round
        errors = vertcat(errors,abs((1/length(round_errors)) * sum(round_errors(1,:))));

        % Divide by num of samples
        update_wKM = (1/length(X)) .* update_wKM;
        update_wMD = (1/length(X)) .* update_wMD;

        % Remove bias z0 node.
        update_wMD = update_wMD(2:end,:);

        % Update
        wKM_new = wKM - learningRate * update_wKM;
        wMD_new = wMD - learningRate * update_wMD;

        diffKM = nnz(arrayfun(@abs, update_wKM) > stopDifference);
        diffMD = nnz(arrayfun(@abs, update_wMD) > stopDifference);

        wKM = wKM_new;
        wMD = wMD_new;

        % Predict error on validation set.
        eee = zeros(1,length(X_validate));
        for x=1:length(X_validate)
            [~,~,y_pred] = forwardProp(X_validate(:,x), h, wMD, wKM);
            y_delta = y_pred - y_validate(x);
            eee(1,x) = y_delta ^ 2;
        end
        errors_val = vertcat(errors_val, abs((1/length(eee)) * sum(eee(1,:))));
    end
end

