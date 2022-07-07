def Update_W(D, beta):
    
    n_X = len(D) - 1
    n_Y = len(D[0]) - 1
    
    temp1 = D[0:n_X,0:n_Y] + beta
    temp2 = np.transpose(np.tile(D[0:n_X,n_Y], (n_X,1)))
    temp3 = np.tile(D[n_X,0:n_Y], (n_Y,1))
    temp4 = np.zeros((n_Y,n_X))

    cost = np.block([[temp1, temp2],[temp3, temp4]])
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    row_ind = [n_X if x>n_X else x for x in row_ind]
    col_ind = [n_Y if x>n_Y else x for x in col_ind]
    
    W = np.zeros((n_X + 1, n_Y + 1))
    W[row_ind,col_ind] = 1
    W[n_X,n_Y] = 0
    
    return W
