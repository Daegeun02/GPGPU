import numpy as np
from pycuda.compiler import SourceModule

class GetGradient:
    def __init__(self, shared):
       self.shared = shared
       
       self.kernel_function()

    def run(self):

        self.initialize()

        ## get out = np.dot(A, theta) - b
        self.first(self.shared.GPU_out,
                   self.shared.GPU_A,
                   self.shared.GPU_theta,
                   self.shared.GPU_b,
                   np.int32(self.shared.length),
                   np.int32(self.shared.width),
                   block=(self.shared.TPB,1,1),
                   grid=(self.shared.BPG,1,1))

        ## get grad = np.dot(A.T, out)
        self.second(self.shared.GPU_grad,
                   self.shared.GPU_A,
                   self.shared.GPU_out,
                   np.int32(self.shared.TPB),
                   np.int32(self.shared.BPG),
                   np.int32(self.shared.length),
                   np.int32(self.shared.width),
                   block=(self.shared.BPG,1,1),
                   grid=(self.shared.width,1,1))
                   
                   

    def kernel_function(self):
        ## block=(thread_per_block,1,1), grid=(block_per_grid,1,1)
        first_ker_function = \
        """
        #define x (threadIdx.x + blockIdx.x * blockDim.x)

        __global__ void first(float* out, float* A, float* theta, float* b, int length, int width) {
            
            if (x < length) {
                for (int j = 0; j < width; j++) {
                    int index1 = x * width + j;

                    out[x] += A[index1] * theta[j];
                    }

                out[x] -= b[x];
            }
        }
        """
        first_ker = SourceModule(first_ker_function)



        ## block=(block_per_grid,1,1), grid=(width,1,1)
        second_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)

        __global__ void second(float* grad, float* A, float* out, int thread_per_block, int block_per_grid, int length, int width) {

            __shared__ float grad_jerk[1000];

            grad_jerk[tx] = 0;

            __syncthreads();
            
            for (int i = 0; i < thread_per_block; i++) {
                int index1 = tx * thread_per_block + i;
                int index2 = index1 * width + bx;
                
                grad_jerk[tx] += A[index2] * out[index1];
            }

            __syncthreads();

            if (tx == 0) {
                for (int i = 0; i < block_per_grid; i++) {
                    grad[bx] += grad_jerk[i];
                }
            }
            else {
                grad_jerk[1000-tx] = 0;
            }

            __syncthreads();
        }
        """
        second_ker = SourceModule(second_ker_function)

        self.first = first_ker.get_function("first")
        self.second = second_ker.get_function("second")

    def initialize(self):
        self.shared.GPU_out[:] = self.shared.init_out[:]