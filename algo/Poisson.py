import numpy as np
'''
Formula for Poisson distribution
Inputs:
    n - number of instances(e.g. number of broken details in production)
    p - probability of success for every single instance( of getting broken)
    m - number of results, that meet our requirements(e.g. exactly 2 pieces are broken)
'''


def  poisson(n, p, m):
	lambd = n * p
	fact = np.math.factorial(m)
	answer = (lambd ** m) / fact * np.e ** (-lambd)
	return answer
