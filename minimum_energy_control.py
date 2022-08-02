from kernel_function import KernelFunctions

import pycuda.driver as cuda
import numpy         as np
import math



class MinimumEnergyControl:

    ## define kernel functions
    kernel_functions = KernelFunctions.define_MEC_kernel_functions()

    get_gradient   = kernel_functions["get_gradient"]
    get_G_matrix   = kernel_functions["get_G"]
    get_Q_matrix   = kernel_functions["get_Q"]
    get_C_matrix   = kernel_functions["get_C"]
    get_weight     = kernel_functions["get_weight"]
    get_bias       = kernel_functions["get_bias"]
    get_mva_weight = kernel_functions["get_mva_weight"]
    get_mva_bias   = kernel_functions["get_mva_bias"]

    def __init__(self, x_des, x_0, dt, lambdas):

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
        if lambdas.nbytes == 8:
            self.rho = 1 / lambdas[0]
            
            ## more velocity accuracy
            self.mva = False

        elif lambdas.nbytes == 16:
            lambdas = lambdas.astype(np.float32)
            lambdas_byte = lambdas.nbytes
            self.lambdas = cuda.mem_alloc(lambdas_byte)
            cuda.memcpy_htod(self.lambdas, lambdas)

            ## more velocity accuracy
            self.mva = True

        else:
            print("not proper lambdas")
            return ValueError()

################################################################################

    def run(self, step):
        ## get_gradient
        MinimumEnergyControl.get_gradient(
            self.weight,
            self.u,
            self.bias,
            self.iteration,
            self.gradient,
            np.int32(step),
            block=(self.TPB,1,1),
            grid=(self.axis*step,1,1)
        )

################################################################################

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

################################################################################
        
    def define_optimal_kernel_size(self, n):
        thread_per_block = int(math.sqrt(n / 2))
        
        iteration = int(n / thread_per_block) + 1

        return thread_per_block, np.int32(iteration)

################################################################################

    def memory_allocation(self, step):
        if self.mva:
            ## identity matrix: same size as rho_matrix
            identity = (np.identity(self.axis*step)).astype(np.float32)
            identity_byte = identity.nbytes
            self.identity = cuda.mem_alloc(identity_byte)
            cuda.memcpy_htod(self.identity, identity)
        else:
            ## rho matrix: 36 * step * step bytes
            rho_matrix      = (math.sqrt(self.rho) * np.identity(self.axis*step)).astype(np.float32)
            rho_matrix_byte = rho_matrix.nbytes
            self.rho_matrix = cuda.mem_alloc(rho_matrix_byte)
            cuda.memcpy_htod(self.rho_matrix, rho_matrix)

        ## solution!!!
        u      = np.zeros((self.axis*step,1)).astype(np.float32)
        u_byte = u.nbytes
        self.u = cuda.mem_alloc(u_byte)
        cuda.memcpy_htod(self.u, u)

        ## G
        G       = np.zeros((self.DOF*self.axis*step)).astype(np.float32)
        G_byte = G.nbytes
        self.G = cuda.mem_alloc(G_byte)
        cuda.memcpy_htod(self.G, G)

        ## gram_G
        weight     = np.zeros((self.axis*self.axis*step*step)).astype(np.float32)
        weight_byte = weight.nbytes
        self.weight = cuda.mem_alloc(weight_byte)
        cuda.memcpy_htod(self.weight, weight)

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
        bias      = np.zeros((self.axis*step)).astype(np.float32)
        bias_byte = bias.nbytes 
        self.bias = cuda.mem_alloc(bias_byte)
        cuda.memcpy_htod(self.bias, bias)

        ## gradient
        gradient      = np.zeros((self.axis*step)).astype(np.float32)
        gradient_byte = gradient.nbytes
        self.gradient = cuda.mem_alloc(gradient_byte)
        cuda.memcpy_htod(self.gradient, gradient)

################################################################################

    def define_matrix(self, step):
        MinimumEnergyControl.get_G_matrix(
            self.input_matrix,
            self.dt,
            self.G,
            block=(6,1,1),
            grid=(step,1,1)
        )
        
        MinimumEnergyControl.get_Q_matrix(
            self.gravity_matrix,
            self.dt,
            self.Q,
            block=(step,1,1),
            grid=(2,1,1)
        )

        MinimumEnergyControl.get_C_matrix(
            self.x_des,
            self.dt,
            self.x_current,
            self.Q,
            self.C,
            block=(3,1,1),
            grid=(step,1,1)
        )

        if self.mva:
            MinimumEnergyControl.get_mva_weight(
                self.G,
                self.lambdas,
                self.identity,
                self.weight,
                block=(3,2,1),
                grid=(step,step,1)
            )

            MinimumEnergyControl.get_mva_bias(
                self.G,
                self.C,
                self.lambdas,
                self.bias,
                block=(2,1,1),
                grid=(self.axis*step,1,1)
            )

        else:
            MinimumEnergyControl.get_weight(
                self.G,
                self.rho_matrix,
                self.weight,
                block=(3,1,1),
                grid=(step,step,1)
            )
            
            MinimumEnergyControl.get_bias(
                self.G,
                self.C,
                self.bias,
                block=(2,1,1),
                grid=(self.axis*step,1,1)
            )

################################################################################

    def memory_free(self):
        self.rho_matrix.free()
        self.identity.free()
        self.u.free()
        self.G.free()
        self.weight.free()
        self.Q.free()
        self.C.free()
        self.bias.free()
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

################################################################################

    def copy_and_unpack_result(self, step):

        if self.mva:
            pass
            
        else:
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
        weight = np.empty((self.axis*self.axis*step*step)).astype(np.float32)
        cuda.memcpy_dtoh(weight, self.weight)

        ## copy Q matrix
        Q = np.empty((self.DOF)).astype(np.float32)
        cuda.memcpy_dtoh(Q, self.Q)

        ## copy C matrix
        C = np.empty((self.DOF)).astype(np.float32)
        cuda.memcpy_dtoh(C, self.C)

        ## copy G_C matrix
        bias = np.empty((self.axis*step)).astype(np.float32)
        cuda.memcpy_dtoh(bias, self.bias)

        ## copy gradient vector
        gradient = np.empty((self.axis*step)).astype(np.float32)
        cuda.memcpy_dtoh(gradient, self.gradient)

        ## pack data
        matrices = dict()
        # matrices["rho_matrix"] = rho_matrix.reshape(self.axis*step,self.axis*step)
        matrices["u"]          = u.reshape(self.axis*step,1)
        matrices["G"]          = G.reshape(self.axis*step,self.DOF).T 
        matrices["weight"]     = weight.reshape(self.axis*step,self.axis*step) 
        matrices["Q"]          = Q.reshape(self.DOF,1)
        matrices["C"]          = C.reshape(self.DOF,1)
        matrices["bias"]       = bias.reshape(self.axis*step,1)
        matrices["gradient"]   = gradient.reshape(self.axis*step,1)

        ## delete all memory
        self.memory_freeall()

        return matrices