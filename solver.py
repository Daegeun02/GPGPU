from shared                 import Shared
from get_gradient           import GetGradient
from optimizer              import GradientMethod, MomentumMethod, NesterovMethod

from minimum_energy_control import MinimumEnergyControl
from optimizer              import OptimizerForGuidance
from constraints_for_input  import ConstraintsForInput
from evaluate               import Evaluator

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
    def __init__(self, x_des, x_0, upper_boundary, downer_boundary, dt=0.1, step=50, max_epoch=100, max_iteration=20):
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
        self.MEC = MinimumEnergyControl(x_des, x_0, dt)

        ## initialize optimizer
        learning_rate = 1e-4

        self.optimizer = OptimizerForGuidance(self.MEC, learning_rate, self.step)

        ## constraint
        self.upper_boundary  = upper_boundary
        self.downer_boundary = downer_boundary

        self.constraint = ConstraintsForInput(self.MEC, self.upper_boundary, self.downer_boundary)

        ## evaluate
        self.evaluator = Evaluator(self.MEC, self.optimizer)
        self.error = 0

        ## initial kernel size
        self.TPB = int(math.sqrt(step))
        self.iteration = int(math.sqrt(step))
        
################################################################################

    def solve(self):
        ##define problem: fit matrices for left step
        self.define_problem()

        ## iteration
        epoch = 0

        while (epoch < self.max_epoch):
            ## initialize
            iteration = 0

            ## learning
            while (iteration < self.max_iteration):
                ## get gradient
                self.MEC.run(self.step)

                ## optimize
                self.optimizer.run(self.step)

                ## tune learning rate
                error = self.evaluator.evaluate_error(self.error,
                                                      self.iteration,
                                                      self.step,
                                                      self.TPB)

                self.error = error

                iteration += 1

            ## constraint
            self.constraint.projection(self.step)

            ## evaluate gradient
            value = self.evaluator.evaluate_gradient(self.step)

            if value:
                break

            else:
                pass

            ## update
            epoch += 1

            ## free memory
            # self.memory_free()

        ## unpack opt_u, other variables
        matrices = self.copy_and_unpack_result()

        return matrices["u"], matrices

################################################################################

    def define_problem(self):
        ## define problem
        self.MEC.define_problem(self.step)

        ## define error vector
        self.evaluator.define_error_vector(self.step)

        ## kernel size
        self.TPB, self.iteration = self.define_optimal_kernel_size(self.axis*self.step)

    def define_optimal_kernel_size(self, n):
        thread_per_block = int(math.sqrt(n / 2))
        
        iteration = int(n / thread_per_block) + 1

        return thread_per_block, np.int32(iteration)

################################################################################

    def memory_free(self):
        self.evaluator.memory_free()

    def memory_freeall(self):

        try:
            self.MEC.memory_freeall()
            self.optimizer.memory_free()
            self.evaluator.memory_free()

        except:
            pass

################################################################################

    def copy_and_unpack_result(self):
        
        ## unpack matrix
        try:
            matrices = self.MEC.copy_and_unpack_result(self.step)
        except:
            matrices = dict()

        ## delete all memory
        self.memory_freeall()

        return matrices