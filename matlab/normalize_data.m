function [Xnorm MINnorm MAXnorm] = normalize_data(X)
% This function takes in the matrix X and normalize the data column wise following the function for normalization (X - MAX)/(MAX - MIN)
  [m n] = size(X);
%  MINnorm = X(1,:);
%  MAXnorm = X(1,:);
%  for i = 1:1:m,
%    if X(i, :) < MINnorm,
%      MINnorm = X(i, :);
%    elseif X(i, :) > MAXnorm,
%      MAXnorm = X(i, :);
%    end
%  end
%  disp(MINnorm);
%  disp(MAXnorm);
  % or use the matlab in-built function max and min
  MINnorm = min(X);
  MAXnorm = max(X);
  for i = 1:1:m,
    for j = 1:1:n,  
      Xnorm(i, j) = (X(i, j)-MINnorm(1, j))/(MAXnorm(1, j) - MINnorm(1, j));
    end
  end 
%  disp(Xnorm);
end
