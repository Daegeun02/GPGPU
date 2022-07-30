from shared                 import Shared
from get_gradient           import GetGradient
from optimizer              import GradientMethod, MomentumMethod, NesterovMethod, OptimizerForGuidance

from minimum_energy_control import MinimumEnergyControl
from optimizer              import OptimizerForGuidance
from constraints_for_input  import ConstraintsForInput

from pycuda.compiler import SourceModule
from pycuda import gpuarray
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import math



class LeastSquare:
    def __init__(self, A, b, learning_rate, beta=0, epoches=10, iteration=5, optimize_method="GD", constrained=None):
        ## shared
        self.shared = Shared(A, b, learning_rate, beta=beta)

        ## gradient
        self.get_gradient = GetGradient(self.shared)
        
        ## optimizer
        if optimize_method == "GD":
            self.optimizer = GradientMethod(self.shared)
        
        elif optimize_method == "momentum":
            self.optimizer = MomentumMethod(self.shared)
            self.shared.momentum(beta)

        elif optimize_method == "Nesterov":
            self.optimizer = NesterovMethod(self.shared)
            self.shared.nesterov(beta)
            
        else:
            return NotImplementedError()

        ## epoches, iteration
        self.epoches = epoches
        self.iteration = iteration

        ## constrained
        if constrained == None:
            pass

        else:
            self.shared.constrained_unpacking(constrained)

        ## error log
        self.error = np.zeros(epoches*iteration)

    def solve(self):
        for epoch in range(self.epoches):
            for iter in range(self.iteration):
                ## get gradient
                self.get_gradient.run()

                ## optimize
                self.optimizer.run()

    def solve_with_record(self):
        for epoch in range(self.epoches):
            for iter in range(self.iteration):
                ## record
                self.record_error(epoch, iter)
                
                ## get gradient
                self.get_gradient.run()

                ## optimize
                self.optimizer.run()

    def record_error(self, epoch, iter):
        index = epoch * self.iteration + iter

        self.get_gradient.initialize()

        self.get_gradient.first(self.shared.GPU_out,
                                self.shared.GPU_A,
                                self.shared.GPU_theta,
                                self.shared.GPU_b,
                                np.int32(self.shared.length),
                                np.int32(self.shared.width),
                                block=(self.shared.TPB,1,1),
                                grid=(self.shared.BPG,1,1))
            
        self.error[index] = np.linalg.norm(self.shared.GPU_out.get())



class MinimumEnergyControlSolver:
    def __init__(self, x_des, x_0, upper_boundary, downer_boundary, step=50, max_epoch=100, max_iteration=20):
        ## important constants
        self.axis = 3
        self.DOF  = 6
        self.initial_step = step

        ## step size
        self.step = step

        ## max epoch
        self.max_epoch = max_epoch
        
        ## max iteration
        self.max_iteration = max_iteration

        ## initialize MEC(minimum energy control)
        self.MEC = MinimumEnergyControl(x_des, x_0)

        ## initialize optimizer
        learning_rate = 1e-4

        self.optimizer = OptimizerForGuidance(learning_rate, self.step)

        ## constraint
        self.upper_boundary  = upper_boundary
        self.downer_boundary = downer_boundary

        self.constraint = ConstraintsForInput(self.MEC, self.upper_boundary, self.downer_boundary)

        ## initial kernel size
        self.TPB = int(math.sqrt(step))
        self.iteration = int(math.sqrt(step))

        ## to compare whether break or not
        self.epsilon = 1e-4

################################################################################

    def solve(self):
        ## define problem: fit matrices for left step
        self.define_problem()

        ## iteration
        epoch = 0

        while (epoch < self.max_epoch):
            ## initialize
            iteration = 0

            ## learning rate tuning
            while (iteration < self.max_iteration):

                ## get gradient
                self.MEC.run(self.step)

                ## optimize
                self.optimizer.run(self.MEC.u, self.MEC.gradient, self.step)

                ## tune learning rate
                self.learning_rate_tuning(iteration)

                iteration += 1

            ## constraint
            self.constraint.projection(self.step)

            ## evaluate
            self.evaluate(epoch)

            ## update...
            epoch += 1

        ## update state
        # self.update_state(step)

        ## record data

        ## get next step

################################################################################

    def define_problem(self):
        ## define_problem
        self.MEC.define_problem(self.step)

        ## evaluate
        ## error_vector
        error_vector      = np.zeros((self.DOF + self.axis*self.step)).astype(np.float32)
        error_vector_byte = error_vector.nbytes
        self.error_vector = cuda.mem_alloc(error_vector_byte)
        cuda.memcpy_htod(self.error_vector, error_vector)

        ## error for record
        error      = np.zeros((self.max_iteration)).astype(np.float32)
        error_byte = error.nbytes
        self.error = cuda.mem_alloc(error_byte)
        cuda.memcpy_htod(self.error, error)

        ## compare error for learning rate tuning
        error_compare      = np.zeros((self.max_iteration)).astype(np.float32)
        error_compare[-1]  = np.float32(1e+6)
        error_compare_byte = error_compare.nbytes
        self.error_compare = cuda.mem_alloc(error_compare_byte)
        cuda.memcpy_htod(self.error_compare, error_compare)

        ## for compare
        ## state record
        # state      = np.zeros((self.DOF*self.step)).astype(np.float32)
        # state_byte = state.nbytes
        # self.state = cuda.mem_alloc(state_byte)
        # cuda.memcpy_htod(self.state, state)

        ## control input
        # input      = np.zeros((self.axis*self.step)).astype(np.float32)
        # input_byte = input.nbytes
        # self.input = cuda.mem_alloc(input_byte)
        # cuda.memcpy_htod(self.input, input)

        ## norm of gradient
        norm_of_gradient      = np.zeros((self.max_epoch)).astype(np.float32)
        norm_of_gradient_byte = norm_of_gradient.nbytes
        self.norm_of_gradient = cuda.mem_alloc(norm_of_gradient_byte)
        cuda.memcpy_htod(self.norm_of_gradient, norm_of_gradient)

        ## kernel function
        self.kernel_function()

        ## kernel size
        self.TPB, self.iteration = self.define_optimal_kernel_size(self.axis * self.step)

################################################################################
    
    def define_optimal_kernel_size(self, n):
        thread_per_block = int(math.sqrt(n / 2))
        
        iteration = int(n / thread_per_block) + 1

        return thread_per_block, np.int32(iteration)

################################################################################

    def learning_rate_tuning(self, iteration):
        ## get error
        self.calculate_error(iteration)

        ## learning rate tuning
        self.optimizer.learning_rate_tuning(self.error_compare,
                                            self.optimizer.lr_set,
                                            np.int32(iteration),
                                            block=(self.axis,1,1),
                                            grid=(self.step,1,1))

    def calculate_error(self, iteration):

        ## set size
        block_size = self.step + 2
        grid_size  = self.axis * self.step + self.DOF

        ## evaluate learning
        self.get_error_vector(self.MEC.G,
                              self.MEC.rho_matrix,
                              self.MEC.u,
                              self.MEC.C,
                              self.iteration,
                              self.error_vector,
                              block=(self.TPB,1,1),
                              grid=(grid_size,1,1))
        
        self.get_vector_norm(self.error_vector,
                             self.error_compare,
                             np.int32(iteration),
                             block=(block_size,1,1),
                             grid=(1,1,1))

################################################################################

    def evaluate(self, epoch):
        ## get norm of gradient
        self.calculate_norm_of_gradient(epoch)

        ## compare with epsilon
        norm_of_gradient = np.zeros((self.max_epoch)).astype(np.float32)
        cuda.memcpy_dtoh(norm_of_gradient, self.norm_of_gradient)

        if norm_of_gradient[epoch] < self.epsilon:
            return True

        else:
            return False

    def calculate_norm_of_gradient(self, epoch):

        ## set size
        block_size = self.step + 2
            
        self.get_vector_norm(self.MEC.gradient,
                             self.norm_of_gradient,
                             np.int32(epoch),
                             block=(block_size,1,1),
                             grid=(1,1,1))

################################################################################

    def update_state(self, step):
        ## update state
        self.get_next_state(self.MEC.x_current,
                            self.MEC.u,
                            self.MEC.dt,
                            self.MEC.gravity_matrix,
                            self.state,
                            np.int32(step),
                            block=(6,1,1),
                            grid=(1,1,1))

################################################################################

    def memory_free(self):
        pass

    def memory_freeall(self):

        try:
            self.MEC.memory_freeall()
            self.optimizer.memory_free()

        except:
            pass

        self.error_vector.free()
        self.error.free()
        self.error_compare.free()
        self.norm_of_gradient.free()
        # self.state.free()

################################################################################

    def kernel_function(self):
        ## block=(TPB,1,1), grid=(DOF+axis*step,1,1)
        get_error_vector_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)
        #define bs (blockDim.x)
        #define gs (gridDim.x)

        __global__ void get_error_vector(float* G, float* rho_matrix, float* u, float* C, int iteration, float* error_vector) {
            
            if (bx < 6) {
                __shared__ float value[1000];

                value[tx] = 0.0;

                __syncthreads();

                for (int i = 0; i <iteration; i++) {
                    int index1 = i + tx * iteration;
                    int index2 = index1 * 6 + bx;

                    value[tx] += G[index2] * u[index1];
                }

                __syncthreads();

                if (tx == 0) {
                    value[1000] = 0.0;

                    for (int j = 0; j < bs; j++) {
                        value[1000] += value[j];
                    }

                    error_vector[bx] = value[1000] - C[bx];
                }
            }
            else {
                if (tx == 0) {
                    int index1 = bx - 6;
                    int index2 = gs - 5;
                    int index3 = index1 * index2;

                    error_vector[bx] = rho_matrix[index3] * u[index1];
                }

                __syncthreads();
            }
        }
        """
        get_error_vector_ker = SourceModule(get_error_vector_ker_function)

        ## block=(step+2,1,1), grid=(1,1,1)
        get_vector_norm_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)
        #define bs (blockDim.x)

        __device__ float square_root(float value) {
            float s = 0;
            float t = 0;

            s = value / 2;

            for (;s != t;) {
                t = s;
                s = ((value/t) + t) / 2;
            }

            return s;
        }

        __device__ float get_norm(float* vector, int length) {
            float value = 0.0;
            float norm;

            for (int i = 0; i < length; i++) {
                value += vector[i] * vector[i];
            }

            norm = square_root(value);

            return norm;
        }

        __global__ void get_vector_norm(float* vector, float* vector_norm, int index) {

            __shared__ float value[1000];

            int index1 = tx * 3;

            for (int i = 0; i < 3; i++) {
                value[index1+i] = vector[index1+i];
            }

            __syncthreads();

            if (tx == 0) {
                int length = bs * 3;

                vector_norm[index] = get_norm(value, length);
            }

            __syncthreads();
        }
        """
        get_vector_norm_ker = SourceModule(get_vector_norm_ker_function)

        ## block=(6,1,1), grid=(1,1,1)
        # get_next_state_ker_function = \
        # """
        # #define tx (threadIdx.x)

        # __global__ void get_next_state(float* x, float* u, float dt, float* gravity_matrix, float* state, int step) {

        #     __shared__ float momentum[6];
        #     __shared__ float control[6];
        #     __shared__ float gravity[6];

        #     int index1 = tx + step * 6;
        #     // int index2 = tx + step * 3;
        #     int index3 = tx % 3;

        #     if (tx < 3) {

        #         momentum[tx] = x[tx] + dt * x[tx+3];
        #         control[tx]  = 0.5*dt*dt * u[index3];

        #         if (index3 == 2) {
        #             gravity[tx] = gravity_matrix[tx];
        #         }
        #         else {
        #             gravity[tx] = 0.0;
        #         }

        #         // input[index2] = u[index3]; 
        #     }
        #     else {

        #         momentum[tx] = x[tx];
        #         control[tx]  = dt * u[index3];

        #         if (index3 == 2) {
        #             gravity[tx] = gravity_matrix[tx];
        #         }
        #         else {
        #             gravity[tx] = 0.0;
        #         }
        #     }

        #     __syncthreads();

        #     x[tx] = momentum[tx] + control[tx] + gravity[tx];
        #     state[index1] = x[tx];

        #     __syncthreads();
        # }
        # """
        # get_next_state_ker = SourceModule(get_next_state_ker_function)

        self.get_error_vector     = get_error_vector_ker.get_function("get_error_vector")
        self.get_vector_norm            = get_vector_norm_ker.get_function("get_vector_norm")
        # self.get_next_state     = get_next_state_ker.get_function("get_next_state")

################################################################################

    def copy_and_unpack_result(self):
        
        ## unpack matrix
        try:
            matrices = self.MEC.copy_and_unpack_result(self.step)
        except:
            matrices = dict()

        ## copy error
        error = np.empty((self.max_iteration)).astype(np.float32)
        cuda.memcpy_dtoh(error, self.error)

        ## copy error_vector
        error_vector = np.empty((self.DOF + self.axis * self.step)).astype(np.float32)
        cuda.memcpy_dtoh(error_vector, self.error_vector)

        ## copy norm of gradient
        norm_of_gradient = np.empty((self.max_epoch)).astype(np.float32)
        cuda.memcpy_dtoh(norm_of_gradient, self.norm_of_gradient)

        ## copy state
        # state = np.empty((self.DOF*self.initial_step)).astype(np.float32)
        # cuda.memcpy_dtoh(state, self.state)

        ## copy input
        # input= np.empty((self.axis*self.initial_step)).astype(np.float32)
        # cuda.memcpy_dtoh(input, self.input)

        ## pack data
        matrices["error"] = error.reshape(self.max_iteration)
        matrices["error_vector"] = error_vector.reshape(self.DOF + self.axis*self.step,1)
        matrices["norm_of_gradient"] = norm_of_gradient.reshape(1, self.max_epoch)
        # matrices["state"] = state.reshape(self.initial_step,self.DOF).T
        # matrices["input"] = input.reshape(self.initial_step,self.axis).T

        ## delete all memory
        self.memory_freeall()

        return matrices