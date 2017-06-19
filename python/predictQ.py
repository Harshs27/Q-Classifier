#!/usr/bin/env python

# This code uses the trained parameters of the Q-Classifier to predict on the new test input data. 
# -- by Harsh Shrivastava, XRCI, IIT Kharagpur
import ImpactQ, sys, os, pickle
import numpy as np

# the major and minor classes... Important to allocate them correctly as selection of threshold depends on them !

def my_sigmoid(x):
    return 1.0/ (1 + np.exp(-x))

def main():
    # Reading the test and training files.
    if len(sys.argv) < 3:
        print "Usage: enter testFile and trainingFile for the Q-classifier."
        sys.exit(1)

    testfile = sys.argv[1]
    trnfile = sys.argv[2]

    # extracting filename, extensions...
    (dirTest, fileTest) = os.path.split(testfile)
    (fileBasetest, testExtension)=os.path.splitext(testfile)

    (dirTrn, fileTrn) = os.path.split(trnfile)
    (fileBasetrn, trnExtension)=os.path.splitext(trnfile)

    data = np.loadtxt(testfile)
    X = data[:, 1:]
    y = data[:, 0]
    
    # loading the trained parameters of the Q-Classifier
    with open(fileBasetrn+'_param.pickle') as fid:
        MINnorm, MAXnorm, rho, sigma, theta, L, J, th = pickle.load(fid)

    m, n_input = X.shape
    Xnorm = X
    for i in range(0, m):
        for j in range(0, n_input):
            Xnorm[i][j] = (X[i][j] - MINnorm[j])/(MAXnorm[j] - MINnorm[j])

    l, n = L.shape
    F, Sigma, inv_Sigma = ImpactQ.Impact(Xnorm, L, rho, sigma, m, n, l)
    F = np.hstack((np.ones((m, 1), float), F)) # adding a column of 1's to F
    H = np.zeros((m, 1), float)
    Ht = np.zeros((m, 1), float) # threshold values for prediction

    count = 0
    for i in range(0, m):
        if y[i] == 1:
            count = count + 1
    if count < m-count:
        MINOR = 1
    else :
        MINOR = 0

    MAJOR = 1-MINOR

    for i in range(0, m):
        H[i] = my_sigmoid(F[i, :] * np.matrix(theta).transpose())
        print('i = %d and H(i) = %f: y(i)=%d'%( i, H[i], y[i]))
        if H[i] >= th:
            Ht[i] = 1

    misclassified = 0
    X1 = 0.0
    X2 = 0.0
    X3 = 0.0
    X4 = 0.0
    for i in range(0, m):
        if Ht[i] == MINOR and y[i] == MINOR:
            X1 = X1 + 1
        elif Ht[i] == MINOR and y[i] == MAJOR:
            X2 = X2 + 1
        elif Ht[i] == MAJOR and y[i] == MINOR:
            X3 = X3 + 1
        elif Ht[i] == MAJOR and y[i] == MAJOR:
            X4 = X4 + 1
   
    accuracy = float(X1 + X4)/float(X1 + X2 + X3 + X4) * 100.0
    sensitivity = X1/(X1 + X3)
    specificity = X4/(X2 + X4)
    print('The classification table is given below, kindly note that sensitivity and specificity might be interchanged depending on the minor class label\n\n \t y=0 \t y=1 \n H=0\t %d\t%d\n H=1\t%d\t%d\n\n\n Accuracy \tSensitivity \tSpecificity \n%f\t%f\t%f\n'%(X1, X2, X3, X4, accuracy, sensitivity, specificity))
    f = open('Table_'+fileBasetest+'.txt', 'w')
    f.write('The classification table is given below, kindly note that sensitivity and specificity might be interchanged depending on the minor class label\n\n \t y=0 \t y=1 \n H=0\t %d\t%d\n H=1\t%d\t%d\n\n\n  Accuracy \tSensitivity \tSpecificity \n%f\t%f\t%f\n'%(X1, X2, X3, X4, accuracy, sensitivity, specificity))
    f.close()

if __name__ == "__main__":
    main()

