import random
def Random(n, a, b):
    # generates a vector of length n with values randomly distributed from 'a' to 'b'
    x = []
    for i in range(0, n):
        num = a + (b-a)*random.random()
        x.append(num)
    return x
