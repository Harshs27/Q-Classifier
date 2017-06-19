function [F Sigma inv_Sigma] = ImpactQ(X, L, rho, sigma, m, n, l)
  % This function calculates the impact of all the Influence points L on each input point X(i)
  Sigma = zeros(n, n);
  for p=1:1:n,
    for q=1:1:n,
      Sigma(p, q)= rho(p, q) * sigma(p)*sigma(q);
    end
  end
  inv_Sigma = pinv(Sigma);
  F = zeros(m, l);
  for i=1:1:m, % number of training examples
    for j=1:1:l,% Impact from each Influence point
      F(i, j) = exp(-1/(2) * (X(i,:) - L(j,:)) * inv_Sigma * (X(i, :)-L(j, :))' );
    end
  end
end
