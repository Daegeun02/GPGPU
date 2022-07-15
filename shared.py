import numpy as np
from pycuda import gpuarray
import math
import sys

class Shared:
    def __init__(self, A, b, learning_rate, constrained=None):
        ## input matrix's shape
        self.length = A.shape[0]
        self.width = A.shape[1]
        
        ## in CPU
        self.A = np.float32(A)
        self.b = np.float32(b)
        self.theta = np.float32(np.zeros(self.width))
        self.out = np.float32(np.zeros(self.length))
        self.grad = np.float32(np.zeros(self.width))

        ## in GPU
        self.GPU_A = gpuarray.to_gpu(self.A.reshape(self.length*self.width))
        self.GPU_b = gpuarray.to_gpu(self.b)
        self.GPU_theta = gpuarray.to_gpu(self.theta)
        self.GPU_out = gpuarray.to_gpu(self.out)
        self.GPU_grad = gpuarray.to_gpu(self.grad)

        ## for initialize out vector and grad vector
        self.init_out = gpuarray.empty_like(self.GPU_out)
        self.init_grad = gpuarray.empty_like(self.GPU_grad)

        ## TPB: thread_per_block, BPG: block_per_grid        
        self.TPB, self.BPG = self.optimal_block_size(self.length)

        ## learning_rate
        self.learning_rate = learning_rate

        ## for constrained lstsq
        if not constrained:
            pass

        else:
            self.constrained_unpacking(constrained)

    def optimal_block_size(self, n):
        
        thread_per_block = int(math.sqrt(n / 2))

        block_per_grid = int(n / thread_per_block) + 1


        return thread_per_block, block_per_grid

    def constrained_unpacking(self, constrained):
        self.constrained = np.zeros((2,self.width))
        self.constrained[0,:] += sys.float_info.max
        self.constrained[1,:] -= sys.float_info.max

        for i in range(constrained.shape[0]):
            index = constrained[0,i]
            if not constrained[1,i]:
                self.constrained[0,index] = constrained[1,i]
            else:
                pass

            if not constrained[2,i]:
                self.constrained[1,index] = constrained[2,i]
            else:
                pass

        self.constrained = np.float32(self.constrained)

        self.GPU_constrained = gpuarray.to_gpu(self.constrained)

    def momentum(self):
        self.beta = 1/3

    def nesterov(self):
        self.GPU_velocity = gpuarray.empty_like(self.GPU_theta)
        self.beta = 2/3