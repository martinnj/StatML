function [ err ] = rootMeanSq( y_pred, y_true )
    err = sqrt(1/length(y_pred) * sum((y_true - y_pred).^2));
end

