def ADMM_ordinary(X, Y, p, c, mu, rho, max_iter):

    ### Get input and def constants #######################
    D = Create_D(X,Y,p,c)
    switching_penalty = (mu ** p)/2
    n_X = len(X[0,0])
    n_Y = len(Y[0,0])
    T = len(X[0])
    
    tol = 10**-4
    primal_tol = tol*math.sqrt(n_X*n_Y*(T-1)*2) 
    dual_tol = tol*math.sqrt(n_X*n_Y*(T-1)*3) 

    ### Parameters for varing rho ##########################
    rho_incr = 2
    rho_decr = 2
    rho_tol = 10
    
    ### Initilization ######################################
    W, H, S_1, S_2 = Initilization(D, switching_penalty)
    
    lambda_1 = np.zeros((T-1, n_X, n_Y))
    lambda_2 = np.zeros((T-1, n_X, n_Y))
    
    r = np.zeros((T-1, 2, n_X, n_Y))
    
    for iterations in range(max_iter):
        
        ### update W #######################################
        W_prev = np.copy(W)
        delta_lambda = lambda_1[0,:,:] - lambda_2[0,:,:]
        delta_S = S_1[0,:,:] - S_2[0,:,:]
        rho_part = delta_S - 2*W[1,0:n_X,0:n_Y] + np.ones((n_X,n_Y))
        beta = delta_lambda + rho*rho_part
        W[0,:,:] = Update_W(D[0,:,:], beta)
        
        for t in range(1,T-1):
            W_part = - W[t-1,0:n_X,0:n_Y] - W[t+1,0:n_X,0:n_Y] + np.ones((n_X,n_Y))
            rho_part = - delta_S + 2*W_part
            beta = - delta_lambda + rho*rho_part
            delta_lambda = lambda_1[t,:,:] - lambda_2[t,:,:]
            delta_S = S_1[t,:,:] - S_2[t,:,:]
            beta = beta + delta_lambda + rho*delta_S
            W[t,:,:] = Update_W(D[t,:,:], beta)
            
        rho_part = - delta_S - 2*W[T-2,0:n_X,0:n_Y] + np.ones((n_X,n_Y))
        beta = - delta_lambda + rho*rho_part
        W[T-1,:,:] = Update_W(D[T-1,:,:], beta)
            
        ### update H #######################################
        H_prev = np.copy(H)
        H = switching_penalty * np.ones((T-1,n_X,n_Y))
        H = H - lambda_1 - lambda_2 - rho * S_1 - rho * S_2
        H = - H/(2*rho)
        H[H<0] = 0
        
        ### update S_1, S_2, lambda_1 and lambda_2 #########
        S_1_prev = np.copy(S_1)
        S_2_prev = np.copy(S_2)
        for t in range(T-1):
            delta_W = W[t,0:n_X,0:n_Y] - W[t+1,0:n_X,0:n_Y]
            delta_W_H_1 = delta_W - H[t,:,:]
            delta_W_H_2 = - delta_W - H[t,:,:]
            
            ### update S_1 #################################
            min_S_1 = - lambda_1[t,:,:]/rho - delta_W_H_1
            min_S_1[min_S_1<0] = 0
            S_1[t,:,:] = np.copy(min_S_1)

            ### update S_2 #################################
            min_S_2 = - lambda_2[t,:,:]/rho - delta_W_H_2
            min_S_2[min_S_2<0] = 0
            S_2[t,:,:] = np.copy(min_S_2)
            
            ### update lambda_1 ############################
            lambda_1[t,:,:] = lambda_1[t,:,:] + rho*(delta_W_H_1 + S_1[t,:,:])

            ### update lambda_2 ############################
            lambda_2[t,:,:] = lambda_2[t,:,:] + rho*(delta_W_H_2 + S_2[t,:,:])
            
            r[t,0,:,:] = delta_W_H_1 + S_1[t,:,:]
            r[t,1,:,:] = delta_W_H_2 + S_2[t,:,:]
            
        primal_residual = np.linalg.norm(r) 
        z = np.stack((2*W[1:T,0:n_X,0:n_Y], S_1, S_2))
        z_prev = np.stack((2*W_prev[1:T,0:n_X,0:n_Y], S_1_prev, S_2_prev))
        dual_residual = rho*np.linalg.norm((z-z_prev))

        if primal_residual < primal_tol and dual_residual < dual_tol:
            if False:
                print('Number of iteraions: ', iterations+1)
                print('Primal residual: ', primal_residual, 'Primal tolerance: ' , primal_tol)
                print('Dual residual: ', dual_residual, 'Dual tolerance: ' , dual_tol)
                print('W is binary: ', ((W==0) | (W==1)).all())
                # print('r is zero: ', ((r==0)).all())
            break
            
        if primal_residual > rho_tol*dual_residual:
            rho = rho_incr*rho
        elif dual_residual > rho_tol*primal_residual:
            rho = rho/rho_decr
        
    ### Calculate final value #######################################
    objective_value = (D * W).sum() + switching_penalty * H.sum()
    
    return objective_value
