function [tf] = ThQw(Xnorm, L, rho, Ifac, m, n, l, theta, y)
  %code for predicting the outcome of the I2 classifier
  % - Harsh Shrivastava, XRCI, IITKGP
  [F Sigma inv_Sigma] = ImpactQ(Xnorm, L, rho, Ifac, m, n, l);

  F = [ones(m, 1) F];% the dimensions of F = [m (l+1)]
  THRESH = 100; % grid search on threshold
  WEIGHT = 300; % grid search for weighted threshold
  HIGH = 0.5;
  LOW = 0.2;
  % calculate the hypothesis by taking the sigmoid of theta x F
  H = zeros(m ,1);
  Ht = zeros(m ,1);
  Th = zeros(THRESH, WEIGHT); % 
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
  else,
    MINOR = 0;
  end
  MAJOR = 1-MINOR;
  
  for t = 1:1:THRESH,
    for w = 1:1:WEIGHT, 
      wt = (w-1)*(HIGH-LOW)/(WEIGHT - 1) + LOW;
     % wt = (w-1)* 0.2/(WEIGHT - 1) + 0.2; % wt varies from 0.2 to 0.4
      th = t/THRESH;
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
        Th(t, w) = 0;
      else
        val = min(wt*X1/(X1+X3),(1-wt)* X1/(X1+X2));
        Th(t, w) = val;
      end
    end
  end

  for t=1:1:THRESH,
    for w = 1:1:WEIGHT,
      if t == 1 && w==1,
        tf = t/THRESH;
       % wf = (w-1)*0.3/299 + 0.2;
	wf = (w-1)*(HIGH-LOW)/(WEIGHT - 1) + LOW;
     %   wf = (w-1)* 0.2/(WEIGHT - 1) + 0.2; % wt varies from 0.2 to 0.4
        max_val = Th(t, w);
      else,
        if (Th(t, w) > max_val),
          tf = t/THRESH;
          wf = (w-1) * (HIGH-LOW)/(WEIGHT-1) + LOW;
          max_val = Th(t, w);
        end
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
  fprintf('The threshold value selected = %f with weight = %f for max ratio = %f\n', tf, wf, max_val);
  fprintf('The classification table is given below, kindly note that sensitivity and specificity might be interchanged depending on the minor class label\n\n \t y=0 \t y=1 \n H=0\t %d\t%d\n H=1\t%d\t%d\n\n\n Accuracy = %f\nSensitivity = %f\nSpecificity = %f\n',X1, X2, X3, X4, accuracy, sensitivity, specificity);
end
