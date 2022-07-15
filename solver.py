from shared       import Shared
from get_gradient import GetGradient
from optimizer    import GradientMethod, MomentumMethod, NesterovMethod

import numpy as np

class LeastSquare:
    def __init__(self, A, b, learning_rate, epoches=10, iteration=5, optimize_method="GD", constrained=None):
        ## shared
        self.shared = Shared(A, b, learning_rate)

        ## gradient
        self.get_gradient = GetGradient(self.shared)
        
        ## optimizer
        if optimize_method == "GD":
            self.optimizer = GradientMethod(self.shared)
        
        elif optimize_method == "momentum":
            self.optimizer = MomentumMethod(self.shared)
            self.shared.momentum()

        elif optimize_method == "Nesterov":
            self.optimizer = NesterovMethod(self.shared)
            self.shared.nesterov()
            
        else:
            return NotImplementedError()

        ## epoches, iteration
        self.epoches = epoches
        self.iteration = iteration

        ## constrained


        ## error log
        self.error = np.zeros(epoches)

    def solve(self):
        for epoch in range(self.epoches):
            for iter in range(self.iteration):
                ## get gradient
                self.get_gradient.run()

                ## optimize
                self.optimizer.run()
            
            self.record_error(epoch)

    def record_error(self, epoch):
        self.get_gradient.initialize()

        self.get_gradient.first(self.shared.GPU_out,
                                self.shared.GPU_A,
                                self.shared.GPU_theta,
                                self.shared.GPU_b,
                                np.int32(self.shared.length),
                                np.int32(self.shared.width),
                                block=(self.shared.TPB,1,1),
                                grid=(self.shared.BPG,1,1))
            
        self.error[epoch] = np.linalg.norm(self.shared.GPU_out.get())
