import numpy as np
import matplotlib.pyplot as plt
from time import time

## Gram-Schmidt
def GR_algorithm(A):
    Q = np.zeros_like(A)
    Q[0,:] = A[0,:]/np.linalg.norm(A[0,:])
    for i in range(1, A.shape[1]):
        Q[i,:] = A[i,:]

        for j in range(0, i):
            Q[i,:] -= Q[j,:]*np.dot(Q[j,:], A[i,:])    
        
        ## linearly dependence
        if Q[i,:] == np.zeros(A.shape[0]):
            print("it has linearly dependent columns")
            break

        Q[i,:] = Q[i,:] / np.linalg.norm(Q[i,:])
    return Q

## QR factorization
def QR_factorization(A,Q,R):
    norm_of_q = np.linalg.norm(A[:,0])
    Q[:,0] = A[:,0] / norm_of_q
    R[0,0] = norm_of_q

    ## columns
    for i in range(1, A.shape[1]):
        Q[:,i] = A[:,i]
        
        ## rows
        for j in range(i):
            R[j,i] = np.dot(Q[:,j],A[:,i])
            Q[:,i] = Q[:,i] - R[j,i]*Q[:,j]
        norm_of_q = np.linalg.norm(Q[:,i])
        R[i,i] = norm_of_q

        ## linearly dependence
        if norm_of_q == 0:
            print("it has linearly dependent columns")
            break

        Q[:,i] = Q[:,i] / np.linalg.norm(Q[:,i])
    return Q, R

def lstsq(A,Q,R,b):
    
    Q, R = QR_factorization(A,Q,R)
    x_hat = np.linalg.inv(R) @ Q.T @ b

    return x_hat

def get_dagger(A):
    
    Q = np.dot(A.T, A)
    dagger = np.dot(np.linalg.inv(Q), A.T)

    return dagger

def lstsq_with_dagger(A,b):

    dagger = get_dagger(A)
    x_hat = np.dot(dagger, b)
    
    return x_hat

class Test:
    def __init__(self, A, b, data_size=3):
        self.A = A
        self.b = b
        self.data_size = data_size
        self.x_hat_rms = np.zeros((self.data_size))
        self.time_by_data_size = np.zeros((2,self.data_size))
        # data_size is logarithm, base 10, of real data size

    def get_operation_time(self):

        for i in range(self.data_size):
            ## set data
            m = 10**(i+1)
            n = 10**i
            A = np.random.rand(m,n)
            Q = np.zeros((m,n))
            R = np.zeros((n,n))
            b = np.random.rand(m)

            ## get solution x_hat
            ## 1. by np.linalg.lstsq
            start_lstsq = time()
            theta = np.linalg.lstsq(A,b, rcond=None)[0]
            end_lstsq = time()

            ## 2. by ...
            start_QR = time()
            x_hat = lstsq_with_dagger(A,b)
            end_QR = time()

            ## record results
            rms = np.sqrt(np.linalg.norm(theta - x_hat)**2 / n)
            self.x_hat_rms[i] = np.round(rms, 10)
            self.time_by_data_size[0,i] = end_lstsq - start_lstsq
            self.time_by_data_size[1,i] = end_QR - start_QR

    def visualize(self):
        plt.figure(figsize=(16,8))

        plt.subplot(121)
        plt.plot(self.time_by_data_size[0,:], alpha = 0.7, label="by np.linalg.lstsq")
        plt.plot(self.time_by_data_size[1,:], alpha = 0.7, label="by $x = A^{\dagger}b$")
        plt.title("Time spent depending on data size")
        plt.xlabel("data size, $10^{2k+1}$")
        plt.ylabel("Time spent (s)")
        plt.legend()

        plt.subplot(122)
        plt.plot(self.x_hat_rms[:])
        plt.title("root mean sqaure value of x_hat and theta")
        plt.xlabel("data size, $10^{2k+1}$")
        plt.ylabel("RMS value")

        plt.show()
