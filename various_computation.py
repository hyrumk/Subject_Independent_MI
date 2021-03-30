import numpy as np

#<TODO> Fix functions

#entropy
def entropy(X):
    '''

    :param X: (numpy) Input
    :return: (numpy) Entropy of the input
    '''
    unique, count = np.unique(X, return_counts=True, axis=0)
    prob = count/len(X)
    en = np.sum((-1)*prob*np.log2(prob))
    return en

#Joint Entropy
def jEntropy(Y,X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y,X]
    return entropy(YX)

#Conditional Entropy
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jEntropy(Y, X) - entropy(X)

a = np.array([1,2,3,0])
b = np.array([3,4,1,5])
print(jEntropy(a, b))