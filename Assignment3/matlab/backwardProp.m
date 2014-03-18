function [ delta_wMD, delta_wKM, round_error ] = backwardProp( y_pred, y_true, dh, a, wKM, z, sample )
% Get output delta's
            y_delta = y_pred - y_true;
            round_error = y_delta^2;

            % Apply h'() to each a_j
            a_d = arrayfun(dh, a);

            % Get hidden delta's using (5.66) i.e. h'(a) * w_kj * delta_y
            hidden_delta = a_d .* (wKM*y_delta)';

            % Update wKM (See 5.67, 5.27)
            delta_wKM = (y_delta * z');
            % Update wMD (See 5.67, 5.27)
            delta_wMD = (hidden_delta * sample');
end

