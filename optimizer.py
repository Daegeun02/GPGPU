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
    def __init__(self, learning_rate, step):
        ## important parameters
        self.axis = 3
        self.DOF  = 6
        
        ## set parameters
        lr_set = (np.ones((self.axis*step)) * learning_rate).astype(np.float32)
        lr_set_byte = lr_set.nbytes
        self.lr_set = cuda.mem_alloc(lr_set_byte)
        cuda.memcpy_htod(self.lr_set, lr_set)
        
        ## kernel function
        self.kernel_function()

    def run(self, theta, gradient, step):
        ## theta, gradient: gpuarray type variable
        self.basic_optimizer(theta,
                             gradient,
                             self.lr_set,
                             block=(3,1,1),
                             grid=(step,1,1))

    def memory_free(self):
        self.lr_set.free()

    def kernel_function(self):
        ## block=(3,1,1), grid=(step,1,1)
        basic_optimizer_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)

        __global__ void basic_optimizer(float* theta, float* gradient, float* learning_rate) {

            int index = tx + bx * 3;

            theta[index] -= gradient[index] * learning_rate[index];

            __syncthreads();
        }
        """
        basic_optimizer_ker = SourceModule(basic_optimizer_ker_function)

        ## block=(3,1,1), grid=(step,1,1)
        learning_rate_tuning_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)

        __global__ void learning_rate_tuning(float* error_compare, float* learning_rate, int iteration) {

            int index = tx + bx * 3;
            
            if (error_compare[iteration-1] > error_compare[iteration]) {
                learning_rate[index] *= 1.2;
            }
            else {
                learning_rate[index] *= 0.5;
            }

            __syncthreads();
        }
        """
        learning_rate_tuning_ker = SourceModule(learning_rate_tuning_ker_function)

        self.basic_optimizer      = basic_optimizer_ker.get_function("basic_optimizer")
        self.learning_rate_tuning = learning_rate_tuning_ker.get_function("learning_rate_tuning")
        