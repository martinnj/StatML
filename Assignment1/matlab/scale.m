function [ Xout ] = scale( Xbase, Xin )
    Xmean = mean(Xbase);
    Xstd = std(Xbase);
    Xout = Xin;
    for i=1:length(Xbase(1,:))
       Xout(:,i) = Xout(:,i) - Xmean(i);
    end

    for i=1:length(Xbase(1,:))
       Xout(:,i) = Xout(:,i)/Xstd(i);
    end
end