def generate_labels(X):
    """
    @brief      Takes in samples and generates labels.
    @param      X       Samples of sequence data (N x T x D)
    @return     Y, labels of shape (N x T x K)
    """
    # IMPLEMENT ME
    #print(Y)
    g = X.shape[0] #N
    h = X.shape[1] #T
    p = X.shape[2] #D
    labels = np.zeros((g,h,p+1))
    for i in range(g):
        A = X[i].T
        e = A.shape[1]
        f = A.shape[0]
        B = np.zeros((f,e))
        for j in range(e):
            if j<6:
                B[:,j] = A[:,j+1]
            else:
                B[:,j] = np.zeros(f)
        x = np.zeros((1,e))
        x[0,e-1] = 1
        B = np.append(B,x, axis=0)
    #print(B)
        labels[i] = B.T
    #pass
    return labels


def weight_init(fan_in, fan_out):
    """
    @param      fan_in   The number of input units
    @param      fan_out  The number of output units
    @return     The 2d weight matrix initialized using xavier uniform initializer
    """
    # IMPLEMENT ME
    a = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(low=-a, high=a, size=(fan_out, fan_in))


def forward(X):
        """
        @brief      Takes a batch of samples and computes the RNN output
        @param      X   A binary numpy array of shape (N x T x D)
        @return     Hidden states (N x T x H), RNN's output (N x T x K)
        """
        # IMPLEMENT ME
        N,T,D = X.shape
        h_t = np.zeros((N,T,h))
        a_t = np.zeros((N,T,h))
        o_t = np.zeros((N,T,k))
        y_hat = np.zeros((N,T,k))
        for g in range(N):
            #h_nt = h_t[g]
            #a_nt = a_t[g]
            #o_nt = o_t[g]
            x_n = X[g]
            #y_hatn = y_hat[g]
            for i in range(T):
                x_nt = x_n[i]
                #y_hatnt = y_hatn[i]
                if i == 0:
                    a_nt = b + np.matmul(U,x_nt)
                else:
                    a_nt = b + np.matmul(W,h_t[g][i-1].T) + np.matmul(U,x_nt)
                h_t[g][i] = np.tanh(a_nt)
                o_t[g][i] = c + np.matmul(V,h_t[g][i])
                y_hat[g][i] = softmax(o_t[g][i])
        return h_t, y_hat


def backward(X, Y, Y_hat, H, lr):
        """
        @brief      Compute gradients and update weights W, U, b, V, c. You
                    do not need to return anything, modify self.W, self.U,
                    self.b, self.V, self.c
        @param      X       A one-hot numpy array of shape (N x T x D)
        @param      Y       A one-hot numpy array of shape (N x T x K)
        @param      Y_hat   Predictions of shape (N x T x K)
        @param      H       Hidden states of shape (N x T x H)
        @param      lr      Learning rate
        """
        # IMPLEMENT ME
        # dW, db, dc, dU, dV
        dW = np.zeros((W.shape))
        db = np.zeros((b.shape))
        dc = np.zeros((c.shape))
        dU = np.zeros((U.shape))
        dV = np.zeros((V.shape))
        #dh, do, do_t, dh_t
        N,T,D = X.shape
        K = Y.shape[2]
        h = H.shape[2]
        do_t = np.zeros((T,K))
        dh_t = np.zeros((T,h))
        do = np.zeros(K)
        dh = np.zeros(h)
        for i in range(N):
            x_n = X[i]
            y_n = Y[i]
            y_hn = Y_hat[i]
            h_n = H[i]
            for j in range(T-1,-1,-1):
                do_t[j] = y_n[j] - y_hn[j]
                #print(do_t)
                do += y_n[j] - y_hn[j]
                print(do)
                if j == T-1:
                    dh_t[j] = np.matmul(V.T, do_t[j])
                    dh += np.matmul(V.T, do_t[j])
                else:
                    dh_t[j] = np.matmul(V.T, do_t[j]) + np.matmul(W.T, np.matmul(dh_t[j+1],np.diag(1-(h_n[j+1])**2)))
                    dh += np.matmul(V.T, do_t[j]) + np.matmul(W.T, np.matmul(dh_t[j+1],np.diag(1-(h_n[j+1])**2)))
                dc = do
                db += np.matmul(np.diag(1-(h_n[j])**2), dh_t[j])
                dV += np.outer(do_t[j],h_n[j])
                subs = h_n[j-1].reshape(h,1)
                dW += np.diag(1-(h_n[j])**2)*np.matmul(dh_t[j],subs)
                subs2 = x_n[j].reshape(D,1)
                dU += np.matmul(np.diag(1-(h_n[j])**2),np.outer(dh_t[j],subs2.T))
                
        return do