def Get_upper_bound(D, switching_penalty):
    
    ############ Get input and def constants ###################################################
    n_X = len(D[0]) - 1
    n_Y = len(D[0,0]) - 1
    T = len(D)
    
    ############ Initilization #################################################################
    W = np.zeros((T, n_X+1, n_Y+1))

    for iterations in range(T):
        
        ############ update W ##################################################################
        W_prev = np.copy(W)
        beta = switching_penalty * (np.ones((n_X, n_Y)) - W_prev[1,0:n_X,0:n_Y] -10**-3) 
        W[0,:,:] = Update_W(D[0,:,:], beta)
        
        for t in range(1,T-1):
            beta = switching_penalty * (np.ones((n_X, n_Y)) - W_prev[t-1,0:n_X,0:n_Y] - W_prev[t+1,0:n_X,0:n_Y] -10**-3)
            W[t,:,:] = Update_W(D[t,:,:], beta)
           
        beta = switching_penalty * (np.ones((n_X, n_Y)) - W_prev[T-2,0:n_X,0:n_Y] -10**-3)
        W[T-1,:,:] = Update_W(D[T-1,:,:], beta)
        
        ############ Calculate subgradients and objective value ################################        
        if (W == W_prev).all():
            break
            
    switches = 0
    for t in range(0,T-1):
        switches += np.absolute(W[t,0:n_X,0:n_Y] - W[t+1,0:n_X,0:n_Y]).sum()

    objective = (D * W).sum() + switching_penalty * switches
        
    return objective
