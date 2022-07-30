from pycuda.compiler import SourceModule

import numpy as np

class ConstraintsForInput:
    def __init__(self, problem, upper_boundary, downer_boundary):
        ## ex> MEC(minimum energy control)
        self.problem = problem

        self.upper_boundary = np.float32(upper_boundary)
        self.downer_boundary = np.float32(downer_boundary)

        ## kernel function
        self.kernel_function()

    def projection(self, step):
        self.project_function(self.problem.u,
                              self.upper_boundary,
                              self.downer_boundary,
                              block=(3,1,1),
                              grid=(step,1,1))

    def kernel_function(self):
        ##block=(3,1,1), grid=(step,1,1)
        projection_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)

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

        __global__ void projection(float* theta, float upper_boundary, float downer_boundary) {

            __shared__ float u[3];
            __shared__ float norm[1];
            __shared__ float value[3];

            int index = tx + bx * 3;

            u[tx] = theta[index];

            __syncthreads();

            if (tx == 0) {
                norm[0] = get_norm(u, 3);
            }

            __syncthreads();

            if ((norm[0] > downer_boundary) && (norm[0] < upper_boundary)) {
                value[tx] = u[tx];
            }
            else {
                value[tx] = u[tx] * upper_boundary / norm[0];
            }

            __syncthreads();

            theta[index] = value[tx];
        }
        """
        projection_ker = SourceModule(projection_ker_function)

        self.project_function = projection_ker.get_function("projection")