# Impact function: normalised form of electric field equation.
# Calculates the impact of the Influence points on the input points
import numpy as np
def Impact(X, L, rho, sigma, m, n, l):
    Sigma = np.zeros((n, n), float)
    for p in range(0, n):
#        if sigma[p] < 1e-6:
#            sigma[p] = 0
        for q in range(0, n):
#            if rho[p][q] < 1e-6:
#                rho[p][q] = 0
#            elif (float(rho[p][q]) > float(1.0001)):
#                print 'Error: as rho value exceeding 1, truncating to 1'
#                print rho[p][q]
#                rho[p][q] = 1
            Sigma[p][q] = rho[p][q] * sigma[p] * sigma[q]
#    print 'rho inside impact'
#    print rho
    F = np.zeros((m, l), float)
    inv_Sigma = np.linalg.pinv(np.matrix(Sigma));
    for i in range(0, m):#0 to m
        for j in range(0, l):#0 to l
            F[i][j] = np.exp(-1.0/2.0 * np.matrix(X[i, :] - L[j, :]) *inv_Sigma * np.matrix(X[i, :] - L[j, :]).transpose() )
#            if F[i][j] < 1e-6:
#                F[i][j] = 0
    return F, Sigma, inv_Sigma
