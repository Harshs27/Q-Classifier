function [theta sigma rho J] = trainQ(F, X, L, y, rho, sigma, m, n, l, theta, iter1, alpha_theta, alpha_sigma, alpha_rho, MINnorm, MAXnorm, name, J_thresh)
    [F Sigma] = ImpactQ(X, L, rho, sigma, m, n, l);
%   size(F) = [m L]
%    disp(F);

    fprintf('\n The Q classifier is training... \n');
    % Add intercept term
    F = [ones(m, 1) F];    
    J = zeros(iter1, 1);
    grad_sigma = zeros(n, 1);
    grad_rho = zeros(n, n);
    STOP = 0;% flag for stopping the iterations
    for inc = 1:1:iter1,
      if(STOP==0),
          
          % after each iteration the gradient factor is set to zero
          grad_sigma = zeros(n, 1);
          grad_rho = zeros(n, n);

          % DO THE DIMESION checking again of the grad factors. ....!!! IMP !!!
          if(inc == 1),
            fprintf('\nThe Cost/penalty is printed below\n');
            tic;
            start = tic;
          end
          if(inc == 2),
            time_each_iter = finish;
            fprintf('\nTime remaining for simulation to end\n');
            sectohms( time_each_iter * (iter1 - inc));
          end

          [F(:, 2:end) Sigma tempinv_Sigma] = ImpactQ(X, L, rho, sigma, m, n, l);
          predictions = my_sigmoid(F*theta);
          % size of predictions = [m x 1(one)] 
%          disp(rho);
%          disp(sigma);
%          disp(Sigma);
          fprintf('.');

          if (inc == 1),
            % the cost function or the penalty incurred 
            for i = 1:1:m,
              J(inc) = J(inc) + -1/m * (y(i)*log(predictions(i))+(1-y(i)) * log(1-predictions(i)));  
            end
            J_old = J(inc);
            J_new = J(inc);
            % displaying the cost function 
            fprintf('Cost function = %f\n', J(inc));
          end 

          if(mod(inc, 5)== 0),
            %            fprintf('|');
            if(mod(inc, 20)==0),
                fprintf('\nTime remaining for simulation to end\n');
                sectohms( time_each_iter * (iter1 - inc));
                J_old = J_new;
                % the cost function or the penalty incurred 
                for k = 1:1:m,
                  J(inc) = J(inc) + -1/m * (y(k)*log(predictions(k))+(1-y(k)) * log(1-predictions(k)));
                end
                J_new = J(inc);

                % displaying the cost function 
                fprintf('Cost function = %f\n', J(inc));
                % saving the variables after some set of iterations 
                save(strcat('temp_', name), 'MINnorm', 'MAXnorm','rho', 'sigma', 'theta', 'L', 'J');
                if ((J_old-J_new)<J_thresh),
                  fprintf('Simulation ran for %d number of iterations and difference in J_old - J_new = %f-%f = %f\n', inc, J_old, J_new, J_old-J_new);
                  STOP = 1;
                end
            end
          end
          % ************************************************** gradient descent of sigma **************************************************
%          tempinv_Sigma = zeros(n, n);
%          tempinv_Sigma = pinv(Sigma);
          if (alpha_sigma ~= 0), 
            D = zeros(n, n); % the derivative of Sigma inverse
            % gradient descent for Sigma/sigma
            for j = 1:1:n,% for each feature calculate the gradient of sigma
              % calculating the derivative of Sigma wrt sigma_j
              der_Sigma = zeros(n, n);
              for p = 1:1:n,
                for q = 1:1:n,
                  if p==j && q==j,
                    der_Sigma(p, q) = 2*sigma(j);
                  elseif p==j,
                    der_Sigma(p, q) = rho(p, q)*sigma(q);
                  elseif q==j,
                    der_Sigma(p, q) = rho(p, q)*sigma(p);
                  else,
                    der_Sigma(p, q) = 0; 
                  end 
                end
              end
              D = -1*tempinv_Sigma * der_Sigma * tempinv_Sigma;
  %            D = -1*inv(Sigma) * der_Sigma * inv(Sigma);
              for i = 1:1:m, % iterate over each example
                second_matrix = zeros(l, 1);
                for k = 1:1:l,
                  second_matrix(k) = (X(i, :) - L(k, :))*D*(X(i, :)-L(k, :))'; 
                end
                grad_sigma(j) = grad_sigma(j) + ( y(i)- predictions(i) ) .*F(i, 2:end).*theta(2:end, 1)' * second_matrix ;
              end
            end
          
            % updating the sigma 
            for k = 1:1:n,
              sigma(k) = sigma(k) - alpha_sigma * 1/(2*m) * grad_sigma(k);
            end
          end
          % end: gradient descent for sigma
        
          %********************************************************** gradient descent for rho **************************************************
          if(alpha_rho ~= 0), 
            D_rho = zeros(n, n); % the derivative of Sigma inverse
            % gradient descent for rho
            for j_p = 1:1:n,% for each landmark points calculate the gradient of rho
              for j_q = j_p+1:1:n,
                % calculating the derivative of Sigma wrt rho_p,q
                der_Sigma_rho = zeros(n, n);
                for p = 1:1:n,
                  for q = p+1:1:n,
                    if p==j_p && q==j_q,
                      der_Sigma_rho(p, q) = sigma(p)*sigma(q);
                      der_Sigma_rho(q, p) = der_Sigma_rho(p, q);
                    else,
                      der_Sigma_rho(p, q) = 0;
                      der_Sigma_rho(q, p) = der_Sigma_rho(p, q);
                    end 
                  end
                end
                D_rho = -1 *tempinv_Sigma * der_Sigma_rho * tempinv_Sigma;
  %              D_rho = -1 * inv(Sigma) * der_Sigma_rho * inv(Sigma);
                for i = 1:1:m, % iterate over each example
                  second_matrix = zeros(l, 1);
                  for k = 1:1:l,
                    second_matrix(k) =(X(i, :) - L(k, :))*D_rho*(X(i, :)-L(k, :))'; 
                  end 
                  grad_rho(j_p, j_q) = grad_rho(j_p, j_q) + ( y(i)- predictions(i) ) .*F(i, 2:end).*theta(2:end, 1)' * second_matrix ;
                end
              end
            end
            % updating the rho parameters
            for p = 1:1:n, 
              for q = p+1:1:n, 
                if (p==q),
                  rho(p, q) = 1;
                else,
                  rho(p, q) = rho(p, q) - alpha_rho * (1/(2*m))* grad_rho(p,q);
                  rho(q, p) = rho(p, q);
                end
              end
            end
          end
          % end: gradient descent for rho
          % *********************************** gradient descent of theta ************************************************        
          grad_theta = zeros(l+1, 1);
          for k = 1:1:l+1,
            for i = 1:1:m,
              grad_theta(k) = grad_theta(k) + (-1/m)*(y(i)- predictions(i))* F(i, k); 
            end
          end
          theta = theta - alpha_theta * grad_theta;
          % end: gradient descent for theta
          if (inc == 1),
            finish = toc;
          end
      end
    end
end
