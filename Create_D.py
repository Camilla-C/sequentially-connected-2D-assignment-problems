def Create_D(X,Y,p,c):
    
    n_X = len(X[0,0])
    n_Y = len(Y[0,0])
    T = len(X[0])
    D = np.zeros((T,n_X+1,n_Y+1))
    cp = c ** p
    cp2 = cp/2
    
    for t in range(T):
        for i in range(n_X+1):
            X_exists = True
            if i == n_X:
                X_exists = False
            elif any(np.isnan(X[:,t,i])):
                X_exists = False
            
            for j in range(n_Y+1):
                Y_exists = True
                if j == n_Y:
                    Y_exists = False
                elif any(np.isnan(Y[:,t,j])):
                    Y_exists = False
                
                if X_exists and Y_exists:
                    D[t,i,j] = min(cp, (np.linalg.norm(X[:,t,i] - Y[:,t,j], ord=p)) ** p)
                    
                elif not X_exists and not Y_exists:
                    pass
                    
                else:
                    D[t,i,j] = cp2
    
    return D
