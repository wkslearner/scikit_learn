
from numpy import *

sigma = [[1] for x in range(4)]
print(sigma)

gamma = zeros((2, 3))+2
print(gamma)
print(mat(gamma).I)
print(gamma.__class__)
print(mat(gamma).__class__)