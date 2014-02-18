function [ SigmaTheta ] = rotateCov( SigmaML, angle )
  R = [cosd(angle), -sind(angle) ; sind(angle), cosd(angle)];
  Rinv = [cosd(angle), sind(angle) ; -sind(angle), cosd(angle)];
  SigmaTheta = Rinv * SigmaML * R;
end

