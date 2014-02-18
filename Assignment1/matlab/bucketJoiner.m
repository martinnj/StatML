function [ joinedX, joinedY ] = bucketJoiner( buckets, partsToJoin )
  tmp = buckets(partsToJoin(1),:);
  joinedX = tmp{1};
  joinedY = tmp{2};
  for i = 2:length(partsToJoin);
    tmp = buckets(partsToJoin(i),:);
    joinedX = vertcat(joinedX, tmp{1});
    joinedY = vertcat(joinedY, tmp{2});
  end
end

