# This function trains the Q-Classifier using the modified gradient descent. Returns the trained parameters and the cost function
import time, datetime, pickle, sys
import numpy as np
import ImpactQ
def my_sigmoid(x):
    return 1.0/ (1.0 + np.exp(-x))

def train(F, X, L, y, rho, sigma, m, n, l, theta, iter1, alpha_theta, alpha_sigma, alpha_rho, MINnorm, MAXnorm, fileBaseName, J_thresh):
    print 'Training the Q-Classifier...'
    F, Sigma, temp_inv_Sigma = ImpactQ.Impact(X, L, rho, sigma, m, n, l)
    F = np.hstack((np.ones((m, 1), float), F)) # adding a column of 1's to F
    J = np.zeros((iter1, 1), float)
    STOP = 0 # flag to stop simulations 
    for inc in range(0, iter1):
        if (STOP==0):
            grad_sigma = np.zeros((n, 1), float)
            grad_rho = np.zeros((n, n), float)
            if (inc == 0):
                print ' The penalty incurred(Cost function) is shown below'
                start = time.time()
            if (inc == 1):
                time_each_iter = finish-start
                print 'Time remaining for simulation to end'
                print str(datetime.timedelta(seconds = time_each_iter * (iter1-inc)))
            
            F[:, 1::], Sigma, temp_inv_Sigma = ImpactQ.Impact(X, L, rho, sigma, m, n, l)
#            temp_inv_Sigma = np.linalg.pinv(np.matrix(Sigma))
            # setting up a loop to calculate the predictions for each Input points
            predictions = np.zeros((m, 1), float)
            for i in range(0, m):
                predictions[i] = my_sigmoid(F[i, :] * np.matrix(theta).transpose())
            
    #        print(".|."),
            sys.stdout.write('.')
            
            if (inc == 0):
                for i in range(0, m):
                    J[inc] = J[inc] + -1.0/m * (y[i]*np.log(predictions[i])+ (1-y[i]) * np.log(1-predictions[i]))
                J_old = J[inc]
                J_new = J[inc]
                print("Cost function = %f" %(J[inc]))

            if inc%5 == 0 and inc != 0 :
                if inc%20 == 0 :
                    print 'Time remaining for the simulation to end'
                    print str(datetime.timedelta(seconds = time_each_iter * (iter1-inc)))
                    J_old = J_new
                    for i in range(0, m):
                        J[inc] = J[inc] + -1.0/m * (y[i]*np.log(predictions[i])+ (1-y[i]) * np.log(1-predictions[i]))
                    J_new = J[inc]
                    print("Cost function = %f" %(J[inc]))
                    # saving the parameters after certain amount of iterations.
                    with open(fileBaseName+'_param_temp.pickle', 'w') as ft: # temp variables file
                        pickle.dump([MINnorm, MAXnorm, rho, sigma, theta, L, J], ft)
                    if((J_old-J_new)<J_thresh):
                        print('Simulation ran for %d number of iterations and difference in J_old - J_new = %f-%f = %f\n'%( inc, J_old, J_new, J_old-J_new))
                        STOP = 1
            # Gradient descent of sigma
            if (alpha_sigma != 0):
                D_sigma = np.zeros((n, n), float)
                for j in range(0, n):
                    der_sigma = np.zeros((n, n), float)# partial derivative of Sigma w.r.t. sigma
                    for p in range(0, n):
                        for q in range(0, n):
                            if p==j and q==j:
                                der_sigma[p][q] = 2 * sigma[j]
                            elif p==j:
                                der_sigma[p][q] = rho[p][q] * sigma[q]
                            elif q==j:
                                der_sigma[p][q] = rho[p][q] * sigma[p]
                            else:
                                der_sigma[p][q] = 0
                    D_sigma = -1.0 * temp_inv_Sigma * np.matrix(der_sigma) * temp_inv_Sigma
                    for i in range(0, m):
                        second_matrix = np.zeros((l, 1), float)
        #                print second_matrix
        #                print l
                        for k in range(0, l):
                            second_matrix[k] = np.matrix(X[i, :] - L[k, :]) * np.matrix(D_sigma) * np.matrix(X[i, :] - L[k, :]).transpose()
                        grad_sigma[j] = grad_sigma[j] + (y[i] - predictions[i]) * np.multiply(np.matrix(F[i, 1::]), np.matrix(theta[1::])) * np.matrix(second_matrix)

                for k in range(0, n):
                    sigma[k] = sigma[k] - alpha_sigma * 1.0/(2*m) * grad_sigma[k]
            # ---------------------------------------------end GD of sigma --------------------------------------
            
            # Gradient descent of rho
            if (alpha_rho != 0):
                D_rho = np.zeros((n, n), float)
                for j_p in range(0, n):
                    for j_q in range(j_p+1, n):
                        der_rho = np.zeros((n, n), float)
                        for p in range(0, n):
                            for q in range(p+1, n):
                                if p==j_p and q==j_q:
                                    der_rho[p][q] = sigma[p] * sigma[q]
                                    der_rho[q][p] = der_rho[p][q]
                                else :
                                    der_rho[p][q] = 0
                                    der_rho[q][p] = der_rho[p][q]
                        D_rho = -1.0 * temp_inv_Sigma* np.matrix(der_rho) * temp_inv_Sigma
                        for i in range(0, m):
                            second_matrix = np.zeros((l, 1), float)
                            for k in range(0, l):
                                second_matrix[k] = np.matrix(X[i, :] - L[k, :]) * np.matrix(D_rho) * np.matrix(X[i, :] - L[k, :]).transpose()
                            grad_rho[j_p][j_q] = grad_rho[j_p][j_q] + (y[i] - predictions[i]) * np.multiply(np.matrix(F[i, 1::]), np.matrix(theta[1::])) * np.matrix(second_matrix)

                for p in range(0, n):
                    for q in range(p+1, n):
                        if(p==q):
                            rho[p][q] = 1
                        else:
                            rho[p][q] = rho[p][q] - alpha_rho * (1.0/(2*m))* grad_rho[p][q]
                            rho[q][p] = rho[p][q]
            # ---------------------------------------------end GD of rho ----------------------------------------

            # Gradient descent of theta
            grad_theta = np.zeros((l+1, 1), float)
            for k in range(0, l+1):
                for i in range(0, m):
                    grad_theta[k] = grad_theta[k] + (-1.0/m)* (y[i] - predictions[i]) * F[i][k]
            
            for k in range(0, l+1):
                theta[k] = theta[k] - alpha_theta * grad_theta[k]
            
            # ---------------------------------------------end GD of theta -----------

            if inc==0:
                finish = time.time()
    
    return theta, sigma, rho, J
