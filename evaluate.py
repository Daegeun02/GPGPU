from kernel_function import KernelFunctions

import pycuda.driver as cuda
import numpy as np



class Evaluator:

    ## define kernel functions
    kernel_functions = KernelFunctions.define_evaluator_kernel_functions()

    get_error_vector     = kernel_functions["get_error_vector"]
    get_mva_error_vector = kernel_functions["get_mva_error_vector"]
    get_vector_norm      = kernel_functions["get_vector_norm"]
    get_norm_of_gradient = kernel_functions["get_norm_of_gradient"]

    def __init__(self, problem, optimizer):
        ## important constants
        self.axis    = 3
        self.DOF     = 6
        self.epsilon = 3

        ## ex> MEC(minimum energy control)
        self.problem = problem

        ## ex> OptimizerForInput
        self.optimizer = optimizer

        ## initialize
        self.error    = np.empty((1)).astype(np.float32)
        self.gradient = np.empty((1)).astype(np.float32)

################################################################################

    def define_error_vector(self, step):

        error_vector      = np.ones((self.axis*step)).astype(np.float32)
        error_vector_byte = error_vector.nbytes
        self.error_vector = cuda.mem_alloc(error_vector_byte)
        cuda.memcpy_htod(self.error_vector, error_vector)

################################################################################

    def evaluate_error(self, pre_error, iteration, step, TPB):
        ## calculate new error(data type: np.float32)
        error = cuda.mem_alloc(4)

        ## get norm of error
        self.calculate_error(error, iteration, step, TPB)

        ## copy error from GPU to CPU
        cuda.memcpy_dtoh(self.error, error)
        error.free()

        ## check we're going good way or not
        ## good way
        if pre_error > self.error[0]:
            self.optimizer.learning_rate *= np.float32(1.2)

        ## bad way
        else:
            self.optimizer.learning_rate *= np.float32(0.5)

        return self.error[0]

    def calculate_error(self, error, iteration, step, TPB):

        ## set size
        block_size = step + 2
        grid_size  = self.axis * step + self.DOF

        ## evaluate learning
        if self.problem.mva:
            Evaluator.get_mva_error_vector(
                self.problem.G,
                self.problem.lambdas,
                self.problem.u,
                self.problem.C,
                iteration,
                self.error_vector,
                block=(TPB,1,1),
                grid=(grid_size,1,1)
            )

        else:
            Evaluator.get_error_vector(
                self.problem.G,
                self.problem.rho_matrix,
                self.problem.u,
                self.problem.C,
                iteration,
                self.error_vector,
                block=(TPB,1,1), 
                grid=(grid_size,1,1)
            )

        Evaluator.get_vector_norm(
            self.error_vector,
            error,
            block=(block_size,1,1),
            grid=(1,1,1)
        )

################################################################################

    def evaluate_gradient(self, step):
        ## calculate new norm of gradient(data type: np.float32)
        gradient = cuda.mem_alloc(4)

        ## get norm of gradient
        Evaluator.get_norm_of_gradient(
            self.problem.gradient,
            gradient    ,
            block=(step,1,1),
            grid=(1,1,1)
        )

        ## copy norm of gradient from GPU to CPU
        cuda.memcpy_dtoh(self.gradient, gradient)
        gradient.free()

        ## compare with epsilon(standard)
        if self.gradient[0] < self.epsilon:
            return True

        else:
            return False

################################################################################

    def memory_free(self):
        self.error_vector.free()
        