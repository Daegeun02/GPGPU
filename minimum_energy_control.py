import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import math



class MinimumEnergyControl:
    def __init__(self, x_des, x_0, dt=0.1):

        ## very important constants
        self.axis = 3
        self.DOF  = 6

        ## gravity, criterion: moon
        gravity = 1.62      # N/kg

        ## A
        state_transition_matrix = \
        np.array([[ 1, 0, 0,dt, 0, 0],
                  [ 0, 1, 0, 0,dt, 0],
                  [ 0, 0, 1, 0, 0,dt],
                  [ 0, 0, 0, 1, 0, 0],
                  [ 0, 0, 0, 0, 1, 0],
                  [ 0, 0, 0, 0, 0, 1]])

        ## B
        input_matrix = \
        np.array([[0.5*dt*dt,        0,        0],
                  [        0,0.5*dt*dt,        0],
                  [        0,        0,0.5*dt*dt],
                  [        dt,       0,        0],
                  [        0,        dt,       0],
                  [        0,        0,       dt]])

        self.input_matrix = cuda.mem_alloc(4*2)
        cuda.memcpy_htod(self.input_matrix, input_matrix[::3,0].astype(np.float32))

        ## g
        gravity_matrix = \
        np.array([[                0],
                  [                0],
                  [0.5*gravity*dt*dt],
                  [                0],
                  [                0],
                  [       gravity*dt]])

        self.gravity_matrix = cuda.mem_alloc(4*2)
        cuda.memcpy_htod(self.gravity_matrix, gravity_matrix[2::3].astype(np.float32))

        ## desired state: x_des
        self.x_des = cuda.mem_alloc(4*self.DOF)
        cuda.memcpy_htod(self.x_des, x_des.astype(np.float32))

        ## initial state: x_0
        self.x_0 = cuda.mem_alloc(4*self.DOF)
        cuda.memcpy_htod(self.x_0, x_0.astype(np.float32))

        ## current state: x_current
        self.x_current = cuda.mem_alloc(4*self.DOF)
        cuda.memcpy_htod(self.x_current, x_0.astype(np.float32))
        
        ## dt
        self.dt = np.float32(dt)

        ## weight
        self.rho = 0.1

        ## define kernel function
        self.kernel_function()

    def run(self, step):
        ## get_gradient
        self.get_gradient(self.gram_G,
                          self.u,
                          self.G_C,
                          self.iteration,
                          self.gradient,
                          np.int32(step),
                          block=(self.TPB,1,1),
                          grid=(self.axis*step,1,1))

    def define_problem(self, step):
        ## initialize
        try:
            self.memory_free()
        except:
            pass

        ## TPB, iteration
        self.TPB, self.iteration = self.define_optimal_kernel_size(self.axis*step)

        ## matrices
        self.memory_allocation(step)
        self.define_matrix(step)

    def define_optimal_kernel_size(self, n):
        thread_per_block = int(math.sqrt(n / 2))
        
        iteration = int(n / thread_per_block) + 1

        return thread_per_block, np.int32(iteration)

    def memory_allocation(self, step):
        ## rho matrix: 36 * step * step bytes
        rho_matrix      = (math.sqrt(self.rho) * np.identity(self.axis*step)).astype(np.float32)
        rho_matrix_byte = rho_matrix.nbytes
        self.rho_matrix = cuda.mem_alloc(rho_matrix_byte)
        cuda.memcpy_htod(self.rho_matrix, rho_matrix)

        ## solution!!!
        u      = np.zeros((self.axis*step,1)).astype(np.float32)
        u_byte = u.nbytes
        self.u = cuda.mem_alloc(u_byte)

        ## G
        G       = np.zeros((self.DOF*self.axis*step)).astype(np.float32)
        G_byte = G.nbytes
        self.G = cuda.mem_alloc(G_byte)
        cuda.memcpy_htod(self.G, G)

        ## gram_G
        gram_G      = np.zeros((self.axis*self.axis*step*step)).astype(np.float32)
        gram_G_byte = gram_G.nbytes
        self.gram_G = cuda.mem_alloc(gram_G_byte)
        cuda.memcpy_htod(self.gram_G, gram_G)

        ## Q
        Q      = np.zeros((self.DOF)).astype(np.float32)
        Q_byte = Q.nbytes
        self.Q = cuda.mem_alloc(Q_byte)
        cuda.memcpy_htod(self.Q, Q)

        ## C
        C      = np.zeros((self.DOF)).astype(np.float32)
        C_byte = C.nbytes
        self.C = cuda.mem_alloc(C_byte)
        cuda.memcpy_htod(self.C, C)

        ## G_C
        G_C      = np.zeros((self.axis*step)).astype(np.float32)
        G_C_byte = G_C.nbytes 
        self.G_C = cuda.mem_alloc(G_C_byte)
        cuda.memcpy_htod(self.G_C, G_C)

        ## gradient
        gradient      = np.zeros((self.axis*step)).astype(np.float32)
        gradient_byte = gradient.nbytes
        self.gradient = cuda.mem_alloc(gradient_byte)
        cuda.memcpy_htod(self.gradient, gradient)

    def define_matrix(self, step):
        self.get_G_matrix(self.input_matrix,
                          self.dt,
                          self.G,
                          block=(6,1,1),
                          grid=(step,1,1))
        
        self.get_Q_matrix(self.gravity_matrix,
                          self.dt,
                          self.Q,
                          block=(step,1,1),
                          grid=(2,1,1))
        
        self.get_G_gram_matrix(self.G,
                               self.rho_matrix,
                               self.gram_G,
                               np.int32(step),
                               block=(3,1,1),
                               grid=(step,step,1))
                               
        self.get_G_C_matrix(self.G,
                            self.x_des,
                            self.dt,
                            self.x_0,
                            self.Q,
                            self.C,
                            self.G_C,
                            block=(3,1,1),
                            grid=(step,1,1))

        ## initialize u
        # G = np.empty((self.DOF*self.axis*step)).astype(np.float32)
        # cuda.memcpy_dtoh(G, self.G)
        # G = G.reshape(self.axis*step,self.DOF).T
        
        # C = np.empty((self.DOF)).astype(np.float32)
        # cuda.memcpy_dtoh(C, self.C)
        # C = C.reshape(self.DOF,1)

        # opt_u = np.linalg.lstsq(G, C, rcond=None)[0]
        # opt_u = np.round(opt_u, 8).astype(np.float32)

        opt_u = np.zeros((self.axis*step,1)).astype(np.float32)
        cuda.memcpy_htod(self.u, opt_u)
        

    def memory_free(self):
        self.rho_matrix.free()
        self.u.free()
        self.G.free()
        self.gram_G.free()
        self.Q.free()
        self.C.free()
        self.G_C.free()
        self.gradient.free()

    def memory_freeall(self):

        try:
            self.memory_free()
        except:
            pass

        self.input_matrix.free()
        self.gravity_matrix.free()
        self.x_des.free()
        self.x_0.free()
        self.x_current.free()

    def kernel_function(self):
        ## block=(TPB,1,1), grid=(axis*step,1,1)
        get_gradient_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)
        #define bs (blockDim.x)
        #define gs (gridDim.x)

        __global__ void get_gradient(float* matrix, float* vector1, float* vector2, int iteration, float* gradient, int step) {

            __shared__ float result[1000];

            result[tx] = 0.0;

            for (int i = 0; i < iteration; i++) {            
                int index1 = i + tx * iteration;
                int index2 = index1 + bx * 3 * step;

                if (index1 < gs) {
                    result[tx] += matrix[index2] * vector1[index1];
                }
                else {
                    result[1000-tx] = 0.0;
                }
            }

            __syncthreads();

            if (tx == 0) {
                gradient[bx] = 0.0;

                for (int j = 0; j < bs; j++) {
                    gradient[bx] += result[j];
                }

                gradient[bx] -= vector2[bx];
            }
            else {
                result[1000-tx] = 0.0;
            }

            __syncthreads();
        }
        """
        get_gradient_ker = SourceModule(get_gradient_ker_function)

        ## block=(6,1,1), grid=(step,1,1)
        get_G_matrix_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)
        #define step (gridDim.x)

        __global__ void get_G_matrix(float* input_matrix, float dt, float* G) {
            // 6: DOF, 18: axis * DOF
            int index = tx + (tx%3) * 6 + bx * 18;

            if (tx < 3) {
                float value;
                value = input_matrix[0] + (step - bx - 1) * dt * input_matrix[1];

                G[index] = value;
            }
            else {
                G[index] = dt;
            }

            __syncthreads();
        }
        """
        get_G_matrix_ker = SourceModule(get_G_matrix_ker_function)

        ## block=(step,1,1), grid=(2,1,1)
        get_Q_matrix_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)
        #define step (blockDim.x)

        __global__ void get_Q_matrix(float* gravity, float dt, float* Q) {
            
            __shared__ float value[1000];

            if (bx == 0) {
                value[tx] = gravity[0] + (tx * dt) * gravity[1];
            }
            else {
                value[tx] = gravity[1];
            }

            __syncthreads();

            if (bx == 0) {
                if (tx == 0) {
                    for (int i = 0; i < step; i++) {
                        Q[2] += value[i];
                    }
                }
            }
            else {
                if (tx == 0) {
                    for (int i = 0; i < step; i++) {
                        Q[5] += value[i];
                    }
                }
            }

            __syncthreads();
        }
        """
        get_Q_matrix_ker = SourceModule(get_Q_matrix_ker_function)

        ## block=(3,1,1), grid=(step,step,1)
        get_G_gram_matrix_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)
        #define by (blockIdx.y)
        #define step (gridDim.x)

        __global__ void get_G_gram_matrix(float* G, float* rho_matrix, float* gram_G) {
            // 3: axis
            int index1 = 3 * step + 1;
            int index2 = 3 * 3 * step;
            int index3 = tx * index1 + bx * 3 + by * index2;

            // 7: DOF+1, 18: axis*DOF
            int index4 = tx * 7 + bx * 18;
            int index5 = tx * 7 + by * 18;

            float value = 0.0;
            value = G[index4] * G[index5] + G[index4+3] * G[index5+3];

            gram_G[index3] = value;

            __syncthreads();

            gram_G[index3] += rho_matrix[index3] * rho_matrix[index3];

            __syncthreads();
        }
        """
        get_G_gram_matrix_ker = SourceModule(get_G_gram_matrix_ker_function)

        ## block=(3,1,1), grid=(step,1,1)
        get_G_C_matrix_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)
        #define step (gridDim.x)

        __global__ void get_G_C_matrix(float* G, float* x_des, float dt, float* x_current, float* Q, float* C, float * G_C) {

            __shared__ float x_A_powered[6];
            __shared__ float C_jerk[6];

            x_A_powered[tx] = x_current[tx] + step * dt * x_current[tx+3];
            x_A_powered[tx+3] = x_current[tx+3];

            __syncthreads();

            C_jerk[tx] = x_des[tx] - Q[tx] - x_A_powered[tx];
            C_jerk[tx+3] = x_des[tx+3] - Q[tx+3] - x_A_powered[tx+3];

            __syncthreads();

            C[tx] = C_jerk[tx];
            C[tx+3] = C_jerk[tx+3];


            __syncthreads();

            // 7: DOF+1, 18: axis*DOF;
            int index1 = tx * 7 + bx * 18;
            int index2 = tx + bx * 3;

            float value;
            value = G[index1] * C_jerk[tx] + G[index1+3] * C_jerk[tx+3];

            __syncthreads();

            G_C[index2] = value;

            __syncthreads();
        }
        """
        get_G_C_matrix_ker = SourceModule(get_G_C_matrix_ker_function)

        self.get_G_matrix      = get_G_matrix_ker.get_function("get_G_matrix")
        self.get_Q_matrix      = get_Q_matrix_ker.get_function("get_Q_matrix")
        self.get_G_gram_matrix = get_G_gram_matrix_ker.get_function("get_G_gram_matrix")
        self.get_G_C_matrix    = get_G_C_matrix_ker.get_function("get_G_C_matrix")
        self.get_gradient      = get_gradient_ker.get_function("get_gradient")

    def copy_and_unpack_result(self, step):
        ## copy rho matrix
        rho_matrix = np.empty((self.axis*self.axis*step*step)).astype(np.float32)
        cuda.memcpy_dtoh(rho_matrix, self.rho_matrix)

        ## copy solution
        u = np.empty((self.axis*step)).astype(np.float32)
        cuda.memcpy_dtoh(u, self.u)

        ## copy G matrix        
        G = np.empty((self.DOF*self.axis*step)).astype(np.float32)
        cuda.memcpy_dtoh(G, self.G)

        ## copy gram matrix of G
        gram_G = np.empty((self.axis*self.axis*step*step)).astype(np.float32)
        cuda.memcpy_dtoh(gram_G, self.gram_G)

        ## copy Q matrix
        Q = np.empty((self.DOF)).astype(np.float32)
        cuda.memcpy_dtoh(Q, self.Q)

        ## copy C matrix
        C = np.empty((self.DOF)).astype(np.float32)
        cuda.memcpy_dtoh(C, self.C)

        ## copy G_C matrix
        G_C = np.empty((self.axis*step)).astype(np.float32)
        cuda.memcpy_dtoh(G_C, self.G_C)

        ## copy gradient vector
        gradient = np.empty((self.axis*step)).astype(np.float32)
        cuda.memcpy_dtoh(gradient, self.gradient)

        ## pack data
        matrices = dict()
        matrices["rho_matrix"] = rho_matrix.reshape(self.axis*step,self.axis*step)
        matrices["u"]          = u.reshape(self.axis*step,1)
        matrices["G"]          = G.reshape(self.axis*step,self.DOF).T 
        matrices["gram_G"]     = gram_G.reshape(self.axis*step,self.axis*step) 
        matrices["Q"]          = Q.reshape(self.DOF,1)
        matrices["C"]          = C.reshape(self.DOF,1)
        matrices["G_C"]        = G_C.reshape(self.axis*step,1)
        matrices["gradient"]   = gradient.reshape(self.axis*step,1)

        ## delete all memory
        self.memory_freeall()

        return matrices