from kernel_function import KernelFunctions

import pycuda.driver as cuda
import numpy as np

from pycuda.compiler import SourceModule



class Optimizer:
    def __init__(self, shared):
        self.shared = shared

    def run(self):
        return NotImplementedError()

    def kernel_function(self):

        ## block=(width,1,1), grid=(1,1,1)
        constrained_projection_ker_function = \
        """
        #define x (threadIdx.x)

        __global__ void constrained_projection (float* theta, float* constrained, int width) {
            int upper = x;
            int downer = x + width;

            if (theta[x] > constrained[upper]){
                if (theta[x] > constrained[downer]) {
                    theta[x] = constrained[upper];
                }
                else {}
            }
            else {
                if (theta[x] < constrained[downer]) {
                    theta[x] = constrained[downer];
                }
                else {}
            }
        }
        """
        constrained_projection_ker = SourceModule(constrained_projection_ker_function)

        self.constrained_projection = constrained_projection_ker.get_function("constrained_projection")

    def initialize(self):
        return NotImplementedError()



class GradientMethod(Optimizer):
    def __init__(self, shared):
        super().__init__(shared)
        super().kernel_function()

        self.kernel_function()

    def run(self):

        self.gradient_method(self.shared.GPU_theta,
                             self.shared.GPU_grad,
                             np.float32(self.shared.learning_rate),
                             block=(self.shared.width,1,1),
                             grid=(1,1,1))

        self.initialize()
        
    def kernel_function(self):

        ## block=(width,1,1), grid=(1,1,1)
        gradient_method_ker_function = \
        """
        #define x (threadIdx.x)

        __global__ void gradient_method (float* theta, float* grad, float learning_rate) {
            theta[x] -= learning_rate * grad[x];

            __syncthreads();
        }
        """
        gradient_method_ker = SourceModule(gradient_method_ker_function)

        self.gradient_method = gradient_method_ker.get_function("gradient_method")

    def initialize(self):
        self.shared.GPU_grad[:] = self.shared.init_grad[:]



class MomentumMethod(Optimizer):
    def __init__(self, shared):
        super().__init__(shared)
        super().kernel_function()

        self.kernel_function()

    def run(self):

        self.momentum_method(self.shared.GPU_theta,
                             self.shared.GPU_grad,
                             np.float32(self.shared.learning_rate),
                             block=(self.shared.width,1,1),
                             grid=(1,1,1))
                             
        self.initialize()

    def kernel_function(self):

        ## block=(width,1,1), grid=(1,1,1)
        momentum_method_ker_function = \
        """
        #define x (threadIdx.x)

        __global__ void momentum_method (float* theta, float* grad, float learning_rate) {
            theta[x] -= learning_rate * grad[x];

            __syncthreads();
        }
        """
        momentum_method_ker = SourceModule(momentum_method_ker_function)
        
        ## block=(width,1,1), grid=(1,1,1)
        momentum_ker_function = \
        """
        #define x (threadIdx.x)

        __global__ void momentum (float* grad, float beta) {

            grad[x] *= beta;
        }
        """
        momentum_ker = SourceModule(momentum_ker_function)

        self.momentum_method = momentum_method_ker.get_function("momentum_method")
        self.momentum = momentum_ker.get_function("momentum")

    def initialize(self):
        self.momentum(self.shared.GPU_grad,
                      np.float32(self.shared.beta),
                      block=(self.shared.width,1,1),
                      grid=(1,1,1))
                      


class NesterovMethod(Optimizer):
    def __init__(self, shared):
        super().__init__(shared)
        super().kernel_function()

        self.kernel_function()
        
        self.iter = 0

    def run(self):

        self.nesterov_method(self.shared.GPU_velocity,
                             self.shared.GPU_theta,
                             self.shared.GPU_grad,
                             np.float32(self.shared.learning_rate),
                             np.float32(self.shared.beta),
                             block=(self.shared.width,1,1),
                             grid=(1,1,1))

        self.initialize()

        self.iter += 1

    def kernel_function(self):

        ## block=(width,1,1), grid=(1,1,1)
        ## theta = y - alpha*grad(y)
        nesterov_method_ker_function = \
        """
        #define x (threadIdx.x)
        __global__ void nesterov_method (float* velocity, float* theta, float* grad, float learning_rate, float beta) {
            
            theta[x] += beta * velocity[x] - learning_rate * grad[x];

            __syncthreads();
        }
        """
        nesterov_method_ker = SourceModule(nesterov_method_ker_function)

        ## block=(width,1,1), grid=(1,1,1)
        nesterov_ker_function = \
        """
        #define x (threadIdx.x)

        __global__ void nesterov (float* theta, float* velocity, float beta) {
            
            theta[x] += beta * velocity[x];
            
            __syncthreads();
        }
        """
        nesterov_ker = SourceModule(nesterov_ker_function)

        self.nesterov_method = nesterov_method_ker.get_function("nesterov_method")
        self.nesterov = nesterov_ker.get_function("nesterov")
        
    def initialize(self):
        
        self.nesterov(self.shared.GPU_theta,
                      self.shared.GPU_velocity,
                      np.float32(self.shared.beta),
                      block=(self.shared.width,1,1),
                      grid=(1,1,1))



class OptimizerForGuidance:

    ## define kernel functions
    kernel_functions = KernelFunctions.define_optimizer_kernel_functions()

    basic_optimizer = kernel_functions["basic_optimizer"]

    def __init__(self, problem, learning_rate):

        ## ex> MEC(minimum energy control)
        self.problem = problem

        ## learning rate
        self.learning_rate = np.float32(learning_rate)
        
################################################################################

    def run(self, step):
        OptimizerForGuidance.basic_optimizer(
            self.problem.u,
            self.problem.gradient,
            self.learning_rate,
            block=(3,1,1),
            grid=(step,1,1)
        )
