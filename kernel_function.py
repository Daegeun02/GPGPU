from pycuda.compiler import SourceModule



class KernelFunctions:
    def __init__(self):
        pass

################################################################################
################################################################################
################################################################################

    @staticmethod
    def define_MEC_kernel_functions():
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

################################################################################

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

################################################################################

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

################################################################################

        ## block=(3,1,1), grid=(step,1,1)
        get_C_matrix_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)
        #define step (gridDim.x)

        __global__ void get_C_matrix(float* x_des, float dt, float* x_current, float* Q, float* C) {

            __shared__ float x_A_powered[6];

            x_A_powered[tx] = x_current[tx] + step * dt * x_current[tx+3];
            x_A_powered[tx+3] = x_current[tx+3];

            __syncthreads();

            C[tx] = x_des[tx] - Q[tx] - x_A_powered[tx];
            C[tx+3] = x_des[tx+3] - Q[tx+3] - x_A_powered[tx+3];

            __syncthreads();
        }
        """
        get_C_matrix_ker = SourceModule(get_C_matrix_ker_function)

################################################################################

        ## block=(3,1,1), grid=(step,step,1)
        get_weight_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)
        #define by (blockIdx.y)
        #define step (gridDim.x)

        __global__ void get_weight(float* G, float* rho_matrix, float* weight) {
            // 3: axis
            int index1 = 3 * step + 1;
            int index2 = 3 * 3 * step;
            int index3 = tx * index1 + bx * 3 + by * index2;

            // 7: DOF+1, 18: axis*DOF
            int index4 = tx * 7 + bx * 18;
            int index5 = tx * 7 + by * 18;

            float value = 0.0;
            value = G[index4] * G[index5] + G[index4+3] * G[index5+3];

            weight[index3] = value;

            __syncthreads();

            weight[index3] += rho_matrix[index3] * rho_matrix[index3];

            __syncthreads();
        }
        """
        get_weight_ker = SourceModule(get_weight_ker_function)

################################################################################

        ## block=(3,2,1), grid=(step,step,1)
        get_mva_weight_ker_function = \
        """
        #define tx (threadIdx.x)
        #define ty (threadIdx.y)
        #define bx (blockIdx.x)
        #define by (blockIdx.y)
        #define gs (gridDim.x)

        __global__ void get_mva_weight(float* G, float* lambdas, float* identity, float* weight) {

            __shared__ float value[6];

            int rank1_x = bx * 18;
            int rank1_y = by * 18;
            
            int index_x = rank1_x + tx * 7;
            int index_y = rank1_y + tx * 7; 

            if (ty == 0) {

                value[2*tx] = lambdas[0] * G[index_x] * G[index_y];
            }
            else {

                value[2*tx+1] = lambdas[1] * G[index_x+3] * G[index_y+3];
            }

            __syncthreads();

            if (ty == 0) {
                int rank_tx = gs * 3 + 1;
                int rank_px = gs * 9;
                int rank_py = 3;

                int index = tx * rank_tx + bx * rank_px + by * rank_py;

                weight[index] = value[2*tx] + value[2*tx+1] + identity[index];
            }

            __syncthreads();
        }
        """
        get_mva_weight_ker = SourceModule(get_mva_weight_ker_function)

################################################################################

        ## block=(2,1,1), grid=(axis*step,1,1)
        get_bias_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)
        #define gs (gridDim.x)

        __global__ void get_bias(float* G, float* C, float* bias) {

            __shared__ float value[2];

            int rank = bx % 3;
            int index = rank + bx * 6;

            if (tx == 0) {
                value[0] = G[index] * C[rank];
            }
            else {
                value[1] = G[index+3] * C[rank+3];
            }

            __syncthreads();

            if (tx == 0) {
                bias[bx] = value[0] + value[1];
            }

            __syncthreads();
        }
        """
        get_bias_ker = SourceModule(get_bias_ker_function)

################################################################################

        ## block=(2,1,1), grid=(axis*step,1,1)
        get_mva_bias_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)
        #define gs (blockDim.x)

        __global__ void get_mva_bias(float* G, float* C, float* lambdas, float* bias) {

            __shared__ float value[2];

            int rank1 = bx % 3;
            int rank2 = bx / 3;

            if (tx == 0) {
                int index = rank1 * 7 + rank2 * 18;

                value[0] = lambdas[0] * G[index] * C[rank1];
            }
            else {
                int index = rank1 * 7 + rank2 * 18;

                value[1] = lambdas[1] * G[index+3] * C[rank1+3];
            }

            __syncthreads();

            if (tx == 0) {
                bias[bx] = value[0] + value[1];
            }

            __syncthreads();
        }
        """
        get_mva_bias_ker = SourceModule(get_mva_bias_ker_function)

################################################################################

        for_MEC = dict()
        for_MEC["get_gradient"]   = get_gradient_ker.get_function("get_gradient")
        for_MEC["get_G"]          = get_G_matrix_ker.get_function("get_G_matrix")
        for_MEC["get_Q"]          = get_Q_matrix_ker.get_function("get_Q_matrix")
        for_MEC["get_C"]          = get_C_matrix_ker.get_function("get_C_matrix")
        for_MEC["get_weight"]     = get_weight_ker.get_function("get_weight")
        for_MEC["get_bias"]       = get_bias_ker.get_function("get_bias")
        for_MEC["get_mva_weight"] = get_mva_weight_ker.get_function("get_mva_weight")
        for_MEC["get_mva_bias"]   = get_mva_bias_ker.get_function("get_mva_bias")

        return for_MEC

################################################################################
################################################################################
################################################################################

    @staticmethod
    def define_optimizer_kernel_functions():
        ## block=(3,1,1), grid=(step,1,1)
        basic_optimizer_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)

        __global__ void basic_optimizer(float* theta, float* gradient, float learning_rate) {

            int index = tx + bx * 3;

            theta[index] -= gradient[index] * learning_rate;

            __syncthreads();
        }
        """
        basic_optimizer_ker = SourceModule(basic_optimizer_ker_function)

################################################################################
        
        for_optimizer = dict()
        for_optimizer["basic_optimizer"] = basic_optimizer_ker.get_function("basic_optimizer")

        return for_optimizer

################################################################################
################################################################################
################################################################################

    @staticmethod
    def define_evaluator_kernel_functions():
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

        ## block=(TPB,1,1), grid=(DOF+axis*step,1,1)
        get_mva_error_vector_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)
        #define bs (blockDim.x)
        #define gs (gridDim.x)

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

        __global__ void get_mva_error_vector(float* G, float* lambdas, float* u, float* C, int iteration, float* error_vector) {
            
            if (bx < 6) {
                __shared__ float value[1000];

                value[tx] = 0.0;

                int rank = bx / 3;

                float rho = square_root(lambdas[rank]);

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

                    error_vector[bx] = (value[1000] - C[bx]) * rho;
                }
            }
            else {
                if (tx == 0) {
                    int index1 = bx - 6;

                    error_vector[bx] = u[index1];
                }

                __syncthreads();
            }
        }
        """
        get_mva_error_vector_ker = SourceModule(get_mva_error_vector_ker_function)


################################################################################

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

################################################################################

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
        
################################################################################

        for_evaluator = dict()
        for_evaluator["get_error_vector"]     = get_error_vector_ker.get_function("get_error_vector")
        for_evaluator["get_mva_error_vector"] = get_mva_error_vector_ker.get_function("get_mva_error_vector")
        for_evaluator["get_vector_norm"]      = get_vector_norm_ker.get_function("get_vector_norm")
        for_evaluator["get_norm_of_gradient"] = get_norm_of_gradient_ker.get_function("get_norm_of_gradient")

        return for_evaluator

################################################################################
################################################################################
################################################################################

    @staticmethod
    def define_constraint_kernel_functions():
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

################################################################################

        for_constraint = dict()
        for_constraint["project_function"] = projection_ker.get_function("projection")

        return for_constraint

################################################################################
################################################################################
################################################################################

    @staticmethod
    def define_stateupdater_kernel_function():
        ## block=(6,1,1), grid=(1,1,1)
        update_state_ker_function = \
        """
        #define tx (threadIdx.x)

        __global__ void update_state(float* transition_matrix, float* input_matrix, float* x_current, float* u, float* gravity_matrix, float* state, int initial_step, int step) {

            __shared__ float transition[6];
            __shared__ float input[6];

            if (tx < 3) {
                transition[tx] = transition_matrix[0] * x_current[tx] + transition_matrix[1] * x_current[tx+3];
                input[tx] = input_matrix[0] * u[tx]; 
            } 
            else {
                transition[tx] = transition_matrix[0] * x_current[tx];
                input[tx] = input_matrix[1] * u[tx];
            }

            __syncthreads();

            int index = step + tx * initial_step;

            state[index] = transition[tx] + input[tx] + gravity_matrix[tx];
        }
        """
        update_state_ker = SourceModule(update_state_ker_function)

################################################################################

        for_updater = dict()
        for_updater["update_state"] = update_state_ker.get_function("update_state")

        return for_updater