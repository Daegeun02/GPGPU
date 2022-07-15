import numpy as np
import matplotlib.pyplot as plt
from time import time 

class LeastSquare():
    def __init__(self, A, b, num_gpu=2, epoches=2):
        self.A = A
        self.b = b
        self.lr = 1e-3/A.shape[1]
        self.num_gpu = num_gpu
        self.epoches = epoches
        ## record each gpu's optimized x
        self.x_hat = np.random.rand(A.shape[1])
        self.x_list = np.zeros((self.num_gpu,self.A.shape[1]))
        self.error_list = []
        self.n = int(self.A.shape[0] / num_gpu)
        self.A1 = A[:self.n,:]
        self.b1 = b[:self.n]
        self.A2 = A[self.n:,:]
        self.b2 = b[self.n:]

    def run(self):        
        for i in range(self.epoches):
            x = self.x_hat

            for j in range(self.num_gpu):
                A, b = self.initialize(j)
                x_ = self.optimize(A, b, x)
                self.x_list[j,:] = x_
                error = self.check(x_)

            self.x_hat = np.sum(self.x_list, axis=0) / self.num_gpu

        return self.x_hat

    ## initialize
    def initialize(self, num_gpu):
        index = np.random.choice(self.n,1000)
        if num_gpu == 0:
            A = self.A1[index,:]
            b = self.b1[index]
        else:
            A = self.A2[index,:]
            b = self.b2[index]

        return A, b

    def optimize(self, A, b, x, iters_per_epoch=500):
        ## optimize x
        for k in range(iters_per_epoch):
            b_ = np.dot(A, x)
            grad = 2 * np.dot(A.T, (b_ - b))
            x -= grad * self.lr

        return x

    def check(self, x):
        b_ = self.A @ x
        error = np.linalg.norm(self.b - b_)
        self.error_list.append(error)

        return error

if __name__ == "__main__":
    A = np.random.rand(10000,1000)
    b = np.random.rand(10000)
    epoch = 20

    t1 = time()
    lstsq = LeastSquare(A,b,epoches=epoch)
    t2 = time()
    dump_time1 = t2 - t1

    t1 = time()
    theta = lstsq.run()
    error = lstsq.check(theta)
    t2 = time()
    calculation_time = t2 - t1
    
    t1 = time()
    x = np.linalg.lstsq(A, b ,rcond=None)[0]
    lstsq_error = np.linalg.norm(lstsq.A @ x - lstsq.b)
    t2 = time()
    lstsq_time = t2 - t1

    t1 = time()
    result = open("data_parallel_result_2.txt", "w")
    result.write(f"error: {error}")
    result.write("\n")
    result.write(f"lstsq error: {lstsq_error}")
    result.write("\n")
    result.write(f"GPU1 error: {lstsq.error_list[-2]}")
    result.write("\n")
    result.write(f"GPU2 error: {lstsq.error_list[-1]}")
    result.write("\n")
    result.write(f"optimal x: {theta}")
    result.close()
    t2 = time()
    dump_time2 = t2 - t1

    t1 = time()
    fig = plt.figure(figsize=(16,8))
    plt.subplot(121)
    plt.plot(lstsq.error_list[::2])
    plt.xlabel("epoches")
    plt.ylabel("error")
    plt.subplot(122)
    plt.plot(lstsq.error_list[1::2])
    plt.xlabel("epoches")
    plt.ylabel("error")
    plt.savefig("data_parallel_error.png", dpi=fig.dpi)
    t2 = time()
    dump_time3 = t2 - t1

    dump_time = dump_time1 + dump_time2 + dump_time3

    print(f"It took {calculation_time} seconds to calculate the least square probelm.")
    print(f"It took {dump_time} seconds to something else.")
    print(f"It took {lstsq_time} seconds to calculate the np.linalg.lstsq.")
    print(f"rms between x and theta: {np.linalg.norm(x - theta)}")