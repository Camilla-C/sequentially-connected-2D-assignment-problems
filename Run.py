import numpy as np
import scipy.io

import math
from scipy.optimize import linear_sum_assignment
import timeit

data = ['7_8_100', '11_10_100', '11_12_100']

p = 1
c = 20
mu = 2
rho = 20
max_iter = 1000

lower_bound = np.zeros((len(data)))
upper_bound = np.zeros((len(data)))
objective_value = np.zeros((len(data)))
di = 0

for dat in data:
    X, Y = Read_data(dat)
    print('Data: ', dat, '\n')
    
    lower_bound[di], upper_bound[di] = Subgrad_Opt2(X, Y, p, c, mu, max_iter)        
    objective_value[di] = ADMM_ordinary(X, Y, p, c, mu, rho, max_iter)
        
    di += 1
    print('-------------------------------------------------------------')
