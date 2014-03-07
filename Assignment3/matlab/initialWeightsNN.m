function [ wMD, wKM ] = initialWeightsNN( K,M,D )
    row = zeros(1,D+1);
    row(1) = 1;
    wMD = vertcat(row,random('Normal',0,1,M,D+1));
    wKM = random('Normal',0,1,K,M+1);
end