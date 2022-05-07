import numpy as np
import matplotlib.pyplot as plt
from time import time 

class LeastSquare():
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.lr = 1e-3/A.shape[1]
        self.x_list = np.zeros((100,A.shape[1]), dtype=np.float32)
        self.error_list = []

    def do(self):
        iters_num = 20
        self.n = int(self.A.shape[0] / iters_num)
        for i in range(iters_num):
            ## initialize
            n = self.n
            A = self.A[i*n:i*n+n,:]
            b = self.b[i*n:i*n+n]
            x = np.random.rand(self.A.shape[1])

            ## optimize x
            ## range of j is not essential for convergence...
            for j in range(500):
                b_ = np.dot(A, x)
                grad = 2 * np.dot(A.T, (b_ - b))
                x -= grad * self.lr

            self.x_list[i,:] = x
            self.error_list.append(self.check(A, b, x))

        self.final_check()

        return x

    def check(self, A, b, x):
        b_ = A @ x
        error = np.linalg.norm(b - b_)

        return error

    def final_check(self):
        x_ = np.sum(self.x_list, axis=0) / self.n
        self.error_list.append(self.check(self.A, self.b, x_))


if __name__ == "__main__":
    A = np.random.rand(10000,1000)
    b = np.random.rand(10000)

    t1 = time()
    lstsq = LeastSquare(A,b)
    t2 = time()
    dump_time1 = t2 - t1

    t1 = time()
    theta = lstsq.do()
    t2 = time()
    calculation_time = t2 - t1

    t1 = time()
    result = open("lstsq_result_cpu1.txt", "w")
    result.write(f"error: {lstsq.error_list[-1]}")
    result.write("\n")
    result.write(f"optimal x: {theta}")
    result.close()
    t2 = time()
    dump_time2 = t2 - t1

    t1 = time()
    fig = plt.figure(figsize=(8,8))
    plt.plot(lstsq.error_list)
    plt.xlabel("epoches")
    plt.ylabel("error")
    plt.savefig("lstsq_error1.png", dpi=fig.dpi)
    t2 = time()
    dump_time3 = t2 - t1

    dump_time = dump_time1 + dump_time2 + dump_time3

    print(f"It took {calculation_time} seconds to calculate the least square probelm.")
    print(f"It took {dump_time} seconds to something else.")