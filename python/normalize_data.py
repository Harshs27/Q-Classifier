# Function for normalizing the input data X 
# uses (X - Max)/ (Max - Min)
def normx(X, m, n):
    MAXnorm = [float("-inf")]*n
    MINnorm = [float("inf")]*n

    for i in range(m):
	    for j in range(n):
                if float(X[i][j]) > float(MAXnorm[j]):
                    MAXnorm[j] = float(X[i][j])
                if float(X[i][j]) < float(MINnorm[j]):
                    MINnorm[j] = float(X[i][j])
    Xnorm = X
    for i in range(0, m):
        for j in range(0, n):
            Xnorm[i][j] = (X[i][j] - MINnorm[j]) / (MAXnorm[j] - MINnorm[j])

    return Xnorm, MINnorm, MAXnorm
