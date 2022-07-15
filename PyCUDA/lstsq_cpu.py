import numpy as np
import matplotlib.pyplot as plt
from time import time 

class LeastSquare():
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.x = np.random.rand(self.A.shape[1])
        self.lr = 1e-3/A.shape[1]
        self.x_list = []
        self.error_list = []

    def do(self):
        for i in range(100):
            ## initialize
            index = np.random.choice(self.A.shape[0],1000)
            A = self.A[index]
            b = self.b[index]

            ## optimize x
            for j in range(100):
                b_ = np.dot(A, self.x)
                grad = 2 * np.dot(A.T, (b_ - b))
                self.x -= grad * self.lr

            self.x_list.append(self.x)
            self.error_list.append(self.check())

        return self.x

    def check(self):
        b_ = self.A @ self.x
        error = np.linalg.norm(self.b - b_)

        return error

if __name__ == "__main__":
    A = np.random.rand(10000,1000)
    b = np.random.rand(10000)

    t1 = time()
    lstsq = LeastSquare(A,b)
    t2 = time()
    dump_time1 = t2 - t1

    t1 = time()
    theta = lstsq.do()
    error = lstsq.check()
    t2 = time()
    calculation_time = t2 - t1

    t1 = time()
    result = open("lstsq_result_cpu.txt", "w")
    result.write(f"error: {error}")
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
    plt.savefig("lstsq_error.png", dpi=fig.dpi)
    t2 = time()
    dump_time3 = t2 - t1

    dump_time = dump_time1 + dump_time2 + dump_time3

    print(f"It took {calculation_time} seconds to calculate the least square probelm.")
    print(f"It took {dump_time} seconds to something else.")