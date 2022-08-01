from kernel_function import KernelFunctions

import numpy as np



class ConstraintsForInput:

    ## define kernel functions
    kernel_functions = KernelFunctions.define_constraint_kernel_functions()

    project_function = kernel_functions["project_function"]

    def __init__(self, problem, upper_boundary, downer_boundary):
        ## ex> MEC(minimum energy control)
        self.problem = problem

        ## constraints
        self.upper_boundary = np.float32(upper_boundary)
        self.downer_boundary = np.float32(downer_boundary)

################################################################################

    def projection(self, step):
        ConstraintsForInput.project_function(
            self.problem.u,
            self.upper_boundary,
            self.downer_boundary,
            block=(3,1,1),
            grid=(step,1,1)
        )
