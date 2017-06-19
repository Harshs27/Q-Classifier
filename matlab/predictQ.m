function predictQ(testFile, paramFile)
  %code for predicting the outcome of the I2 classifier
  % - Harsh Shrivastava, XRCI, IITKGP
  % testfile = test data, paramFile =  the name of the file in which the parameters are to be imported. ext = the extension of files.
 

  data = load(testFile);
  X = data(:, 2:end);
  y = data(:, 1);
  Y = y;
  % extracting the filenames of test and parameter files for saving the results ...
  [p_path, p_name, p_ext] = fileparts(paramFile);
  [t_path, t_name, t_ext] = fileparts(testFile);

  load(strcat(p_name, '_param.mat'));
  [m n_input] = size(X);
  for i = 1:1:m,
    for j = 1:1:n_input,
      Xnorm(i, j) = (X(i, j) - MINnorm(1, j))/(MAXnorm(1, j) - MINnorm(1, j)); % note that the normalisation criteria might change, so remenber to update this!!!!!!!!
    end
  end
  [l n] = size(L); % where l is the number of Influence points and n is the dimension of each Influence point which should be equal to the input x 

  if (n_input ~= n),
    fprintf('error : the dimensions of the input and Influence points do not match... N_input = %d and Influence_N  = %d\n', n_input, n);
  end 

  [F Sigma inv_Sigma] = ImpactQ(Xnorm, L, rho, sigma, m, n, l);
  F = [ones(m, 1) F];% the dimensions of F = [m (l+1)]
  % calculate the hypothesis by taking the sigmoid of theta x F
  H = zeros(m ,1);
  Ht = zeros(m ,1);
  count = 0;
  for i = 1:1:m,
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

  for i = 1:1:m,
    H(i) = my_sigmoid(F(i, :)*theta);
    % setting the threshold for classification to the center meanwhile.
    fprintf('i = %d and H(i) = %f: y(i)=%d\n', i, H(i), y(i));
    if (H(i) >= th),
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
  fprintf('The classification table is given below, kindly note that sensitivity and specificity might be interchanged depending on the minor class label\n\n \t y=0 \t y=1 \n H=0\t %d\t%d\n H=1\t%d\t%d\n\n\n Accuracy \t Sensitivity \t Specificity \n%f\t%f\t%f\n',X1, X2, X3, X4, accuracy, sensitivity, specificity);
 % writing the classification table to an output file... 
  fid = fopen(strcat('Classification_table_', t_name, '.txt'), 'w');
  if fid ~= -1,
    fprintf(fid, 'The classification table is given below, kindly note that sensitivity and specificity might be interchanged depending on the minor class label\n\n \t y=0 \t y=1 \n H=0\t %d\t%d\n H=1\t%d\t%d\n\n\n  Accuracy \t Sensitivity \t Specificity \n%f\t%f\t%f\n',X1, X2, X3, X4, accuracy, sensitivity, specificity);
    fclose(fid);
  end
end
