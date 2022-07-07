def Subgrad_Opt(X, Y, p, c, mu, max_iter):
    
    ############ Get input and def constants ###################################################
    D = Create_D(X,Y,p,c)
    switching_penalty = (mu ** p)/2
    n_X = len(X[0,0])
    n_Y = len(Y[0,0])
    T = len(X[0])

    ############ Initilization #################################################################
    W = np.zeros((T, n_X+1, n_Y+1))
    dual_var = 0.5*switching_penalty*np.ones((T-1, n_X, n_Y))
    delta_W = np.zeros((T-1, n_X, n_Y))
    MDS = 0.5*np.ones((T-1, n_X, n_Y))
    
    ############### CALCULATE UPPER BOUND ######################################################
    upper_bound = Get_upper_bound(D, switching_penalty)
    lower_bound = -np.inf
    
    last_improvement = 0
    alpha = 0.7
    lagrangean_obj = lower_bound
    
    for iterations in range(max_iter):
        
        delta_lambda = 2*dual_var - switching_penalty
        ############ update W ##################################################################
        beta = delta_lambda[0,:,:]
        W[0,:,:] = Update_W(D[0,:,:], beta)
        
        for t in range(1,T-1):
            beta = -delta_lambda[t-1,:,:] + delta_lambda[t,:,:]
            W[t,:,:] = Update_W(D[t,:,:], beta)
           
        beta = - delta_lambda[T-2,:,:]
        W[T-1,:,:] = Update_W(D[T-1,:,:], beta)
        
        for t in range(1,T):
            delta_W[t-1,:,:] = W[t-1,0:n_X,0:n_Y] - W[t,0:n_X,0:n_Y]
        
        ############ Calculate subgradients and objective value ################################
        subgradient = np.copy(delta_W)
        lagrangean_obj_prev = lagrangean_obj
        lagrangean_obj = (D * W).sum() + (delta_lambda*subgradient).sum()
         
        ############ Check special cases for projection ########################################   
        if (dual_var == 0).any() or (dual_var == switching_penalty).any():
            ind1 = np.argwhere(dual_var == 0)
            ind2 = np.argwhere(dual_var == switching_penalty)

            for i in ind1:
                if subgradient[i[0], i[1], i[2]] < 0:
                    subgradient[i[0], i[1], i[2]] = 0
                if MDS[i[0], i[1], i[2]] < 0:
                    MDS[i[0], i[1], i[2]] = 0
                    
            for i in ind2:
                if subgradient[i[0], i[1], i[2]] > 0:
                    subgradient[i[0], i[1], i[2]] = 0  
                if MDS[i[0], i[1], i[2]] > 0:
                    MDS[i[0], i[1], i[2]] = 0  
        
        if (subgradient==0).all():
            lower_bound = lagrangean_obj
            break
        
        if (MDS==0).all():
            if lagrangean_obj > lower_bound:
                lower_bound = lagrangean_obj 
            break
            
        ############ Modified Deflected Subgradient #############################################
        theta = max(0, -(subgradient*MDS).sum() / (np.linalg.norm(subgradient)*np.linalg.norm(MDS)))
        sigma = 1/(2-theta)
        
        gamma_ADS = np.linalg.norm(subgradient)/np.linalg.norm(MDS)
        gamma_MGT = max(0, -sigma * (subgradient*MDS).sum() / (MDS**2).sum())
        gamma_MDS = (1-theta) * gamma_MGT + theta * gamma_ADS
        MDS = subgradient + gamma_MDS * MDS

        step_size = alpha*(upper_bound - lagrangean_obj)/((MDS**2).sum())
        
        dual_var = dual_var + step_size * MDS
        dual_var[dual_var<0] = 0
        dual_var[dual_var>switching_penalty] = switching_penalty
            
        ############### Terminate and update Alpha ####################### 
        if lagrangean_obj > lower_bound:
            prev_lb = lower_bound
            lower_bound = lagrangean_obj 
            if lower_bound - prev_lb < prev_lb*(10**-6):
                small += 1
            else:
                small = 0
                
            last_improvement = 0
            
        if ((upper_bound - lower_bound) < 0.5) or (alpha < 10**-3) or (small == 3): 
            break
            
        if last_improvement >= 20:
            alpha *= 0.5
            last_improvement = 0

        last_improvement += 1
    
    return lower_bound, upper_bound
