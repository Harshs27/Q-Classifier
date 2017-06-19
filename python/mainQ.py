#!/usr/bin/env python

# This code is a part of the Q-classifier: A method of Supervised Classification for Imbalanced Datasets using Influence Fields.
# The mainQ file: Finds the Influence points using k-means on the normalized input data.
#                   Initialises various parameters.
#                   Calls the trainQ function for training the Q-classifier.
# -- by Harsh Shrivastava, XRCI, IIT Kharagpur


import sys, os, re, random, math, pickle
import numpy as np
import normalize_data, gen_rand_vector, ImpactQ, trainQ, ThwQ #<Enter the python files needed to import>
from sklearn.cluster import KMeans

def main():
    # checking to read the input file.
    if len(sys.argv) < 2:
        print "Usage: enter filename for training the Q-classifier."
        sys.exit(1)

    filename = sys.argv[1]
    
    # Initialising constants
    iter1 = 100 # Number of iterations for training the modified grad descent of Q-classifier.
    alpha_theta = 4.4 # learning rate of parameter theta
    alpha_sigma = 0.2 # learning rate of parameter sigma
    alpha_rho = 0.0 # learning rate of parameter rho
    Influence_points = 70 # Number of Influence points to calculate the impact on the input training set
    J_thresh = 0.005 # the threshold for penalty function 'J'
    # NOTE1: Currently not shuffling the data
    data = np.loadtxt(filename)
    X = data[:, 1:]
    y = data[:, 0]
    
    m, n = X.shape
    # Kindly note that: rows = 0 to m-1, cols = 0 to n-1 and the training set is divided into features 'X[rows][cols]' and labels 'y'
    
    # extracting filename, extensions...
    (dirName, fileName) = os.path.split(filename)
    (fileBaseName, fileExtension)=os.path.splitext(fileName)

#    print dirName         # /Users/t/web/perl-python
#    print fileName        # banknote.txt
#    print fileBaseName    # banknote
#    print fileExtension   # .txt
    
    print 'Normalising the input data...'
    Xnorm, MINnorm, MAXnorm = normalize_data.normx(X, m, n)
    Xnorm = np.array(Xnorm)
  
    print 'Finding the Influence points using the K-means cluster centers...'
    kmeans = KMeans(init='k-means++', n_clusters=Influence_points, n_init=10)
    Idx = kmeans.fit_predict(Xnorm) # returns indices for each input data point belonging to their respective clusters.
    L = kmeans.cluster_centers_ # returns the cluster centers.
    print L 
#    print Idx 
    L = np.array(L)
    print 'Influence points initialised...'

    l = len(L) # l=Influence points.. just for cross-checking
#    l = Influence_points 
    print L

    print 'Initialising Theta'
    theta = gen_rand_vector.Random(l+1, float(1/float(l+1)), float(1/float(l+1)))
    theta = np.array(theta)
    print theta
    print 'Initialising sigma by calculating the variance of input data and Influence points.'
    sigma = np.zeros((n, 1), float) # array of zeros 
    mu_L = L.mean(0)
     
    for j in range(0, n):
        for i in range(0, m):
            sigma[j] = sigma[j] + ((Xnorm[i][j] - mu_L[j])**2);
        sigma[j] = math.sqrt(sigma[j] / m)

    print sigma
    
    rho = np.zeros((n, n), float)
    for p in range(0, n):
        for q in range(p, n):
            for i in range(0, m):
                rho[p][q] = rho[p][q] + (Xnorm[i][p] - mu_L[p]) * (Xnorm[i][q] - mu_L[q])
            rho[p][q] = rho[p][q]/(m*sigma[p]*sigma[q])
            rho[q][p] = rho[p][q]
    
    print rho
    Sigma = np.zeros((n, n), float)
    print 'Initialising the Sigma Matrix'
    for p in range(0, n):
        for q in range(0, n):
            Sigma[p][q] = rho[p][q] * sigma[p]* sigma[q]
    
    print Sigma
    
    F, Sigma, inv_Sigma = ImpactQ.Impact(Xnorm, L, rho, sigma, m, n, l) # Calculate the impact of Influence points on the input dataset
    print 'Test Influence calculation done!'
    print Sigma
    print F
    theta, sigma, rho, J = trainQ.train(F, Xnorm, L, y, rho, sigma, m, n, l, theta, iter1, alpha_theta, alpha_sigma, alpha_rho, MINnorm, MAXnorm, fileBaseName, J_thresh)

    print 'Displaying final theta value after running the Q-Classifier'
    print theta
    # Determining the threshold from the training datasets
    print 'Finding the threshold value '
#    th = ThQ.threshold(Xnorm, L, rho, sigma, m, n, l, theta, y)
    th = ThwQ.threshold(Xnorm, L, rho, sigma, m, n, l, theta, y)

    with open(fileBaseName+'_param.pickle', 'w') as fid:
        pickle.dump([MINnorm, MAXnorm, rho, sigma, theta, L, J, th], fid)
        

if __name__ == "__main__":
    main()

