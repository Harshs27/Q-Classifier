function [tf] = ThQ(Xnorm, L, rho, Ifac, m, n, l, theta, y)
  %code for predicting the outcome of the I2 classifier
  % - Harsh Shrivastava, XRCI, IITKGP
  [F Sigma] = ImpactQ(Xnorm, L, rho, Ifac, m, n, l);

  F = [ones(m, 1) F];% the dimensions of F = [m (l+1)]

  % calculate the hypothesis by taking the sigmoid of theta x F
  H = zeros(m ,1);
  Ht = zeros(m ,1);
  Th = zeros(100, 1);
  count = 0;
 
  for i = 1:1:m,
    H(i) = my_sigmoid(F(i, :)*theta);
    if y(i) == 1,
      count = count + 1;
    end
  end
  % assigning the major and minor classes
  if count < m-count,
    MINOR = 1;
    minorCount = count;
  else,
    MINOR = 0;
    minorCount = m-count;
  end
  MAJOR = 1-MINOR;
  majorCount = m - minorCount;
  imbalance = (minorCount)/(majorCount);
  for t = 1:1:100,
    th = t/100.0;
    Ht = zeros(m, 1);
    for i = 1:1:m,
      if H(i) >= th,
        Ht(i) = 1;
      end
    end
    X1 = 0; X2 = 0; X3 =0; X4 = 0;
    for i = 1:1:m,
      if (Ht(i) == MINOR && y(i) == MINOR),
        X1 = X1 + 1;
      elseif (Ht(i) == MINOR && y(i) == MAJOR),
        X2 = X2 + 1;
      elseif (Ht(i) == MAJOR && y(i) == MINOR),
        X3 = X3 + 1;
      elseif (Ht(i) == MAJOR && y(i) == MAJOR),
        X4 = X4 + 1;
      end
    end
    % Decision criterion is maximizing the minimum of recall and precision
    if X1+X3 == 0 or X1+X2 = 0,
      Th(t) = 0;
    else
    % val = min(imbalance * X1/(X1+X3), (1-imbalance) * X1/(X1+X2));
      val = min(0.375 * X1/(X1+X3), (1-0.375) * X1/(X1+X2));
      Th(t) = val;
    end
  end

  for t=1:1:100,
    if t == 1,
      tf = t/100.0;
      max_val = Th(t);
    else,
      if (Th(t) > max_val),
        tf = t/100.0;
        max_val = Th(t);
      end
    end
  end

  Ht = zeros(m, 1);

  for i = 1:1:m,
  % setting the threshold for classification to the center meanwhile.
    fprintf('i = %d and H(i) = %f: y(i)=%d\n', i, H(i), y(i));
    if (H(i) >= tf),
      Ht(i) = 1;
    end
  end

  % check with y (output) if available
  X1 = 0; X2 = 0; X3 =0; X4 = 0;
  for i = 1:1:m,
    if (Ht(i) == MINOR && y(i) == MINOR),
      X1 = X1 + 1;
    elseif (Ht(i) == MINOR && y(i) == MAJOR),
      X2 = X2 + 1;
    elseif (Ht(i) == MAJOR && y(i) == MINOR),
      X3 = X3 + 1;
    elseif (Ht(i) == MAJOR && y(i) == MAJOR),
      X4 = X4 + 1;
    end
  end

  accuracy = (X1 + X4)/(X1 + X2 + X3 + X4) * 100;
  sensitivity = X1/(X1 + X3);
  specificity = X4/(X2 + X4);
  fprintf('The threshold value selected = %f for max ratio = %f\n', tf, max_val);
  fprintf('The classification table is given below, kindly note that sensitivity and specificity might be interchanged depending on the minor class label\n\n \t y=0 \t y=1 \n H=0\t %d\t%d\n H=1\t%d\t%d\n\n\n Accuracy = %f\nSensitivity = %f\nSpecificity = %f\n',X1, X2, X3, X4, accuracy, sensitivity, specificity);
end
