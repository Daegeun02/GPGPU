import pycuda.driver as cuda
import numpy         as np

from pycuda.compiler import SourceModule


class Evaluator:
    def __init__(self):
        ## important constants
        self.axis    = 3
        self.DOF     = 6
        self.epsilon = 1

        ## initialize
        self.error    = np.empty((1)).astype(np.float32)
        self.gradient = np.empty((1)).astype(np.float32)

        ## define kernel function
        self.kernel_function()

################################################################################

    def define_error_vector(self, step):
        error_vector = np.zeros((self.axis*step)).astype(np.float32)
        error_vector_byte = error_vector.nbytes
        self.error_vector = cuda.mem_alloc(error_vector_byte)
        cuda.memcpy_htod(self.error_vector, error_vector)

################################################################################

    def evaluate_error(self, pre_error, problem, optimizer, iteration, step, TPB):
        ## calculate new error(data type: np.float32)
        error = cuda.mem_alloc(4)

        ## get norm of error
        self.calculate_error(problem, error, iteration, step, TPB)

        ## copy error from GPU to CPU
        cuda.memcpy_dtoh(self.error, error)
        error.free()

        ## check we're going good way or not
        ## good way
        if pre_error > self.error[0]:
            optimizer.learning_rate *= np.float32(1.2)

        ## bad way
        else:
            optimizer.learning_rate *= np.float32(0.5)

        return self.error[0]

    def calculate_error(self, problem, error, iteration, step, TPB):

        ## set size
        block_size = step + 2
        grid_size  = self.axis * step + self.DOF

        ## evaluate learning
        self.get_error_vector(problem.G,
                              problem.rho_matrix,
                              problem.u,
                              problem.C,
                              iteration,
                              self.error_vector,
                              block=(TPB,1,1),
                              grid=(grid_size,1,1))
        
        self.get_vector_norm(self.error_vector,
                             error,
                             block=(block_size,1,1),
                             grid=(1,1,1))

################################################################################

    def evaluate_gradient(self, problem, step):
        ## calculate new norm of gradient(data type: np.float32)
        gradient = cuda.mem_alloc(4)

        ## get norm of gradient
        self.get_norm_of_gradient(problem.gradient,
                                  gradient,
                                  block=(step,1,1),
                                  grid=(1,1,1))

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
        ## memory free
        self.error_vector.free()

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

        __global__ void get_vector_norm(float* vector, float* vector_norm) {

            __shared__ float value[1000];

            int index1 = tx * 3;

            for (int i = 0; i < 3; i++) {
                value[index1+i] = vector[index1+i];
            }

            __syncthreads();

            if (tx == 0) {
                int length = bs * 3;

                vector_norm[0] = get_norm(value, length);
            }

            __syncthreads();
        }
        """
        get_vector_norm_ker = SourceModule(get_vector_norm_ker_function)

        ## block=(step,1,1), grid=(1,1,1)
        get_norm_of_gradient_ker_function = \
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

        __global__ void get_norm_of_gradient(float* gradient, float* norm_of_gradient) {

            __shared__ float value[1000];

            int index1 = tx * 3;

            for (int i = 0; i < 3; i++) {
                value[index1+i] = gradient[index1+i];
            }

            __syncthreads();

            if (tx == 0) {
                int length = bs * 3;

                norm_of_gradient[0] = get_norm(value, length);
            }

            __syncthreads();
        }
        """
        get_norm_of_gradient_ker = SourceModule(get_norm_of_gradient_ker_function)


        self.get_error_vector     = get_error_vector_ker.get_function("get_error_vector")
        self.get_vector_norm      = get_vector_norm_ker.get_function("get_vector_norm")
        self.get_norm_of_gradient = get_norm_of_gradient_ker.get_function("get_norm_of_gradient")