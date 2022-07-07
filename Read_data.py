def Read_data(data):
    
    dataX = scipy.io.loadmat('X_' + data + '.mat')
    dataY = scipy.io.loadmat('Y_' + data + '.mat')
    X = dataX['temp']
    Y = dataY['temp']
    
    return X, Y
