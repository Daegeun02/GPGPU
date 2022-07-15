from solver import *
import matplotlib.pyplot as plt

class HyperParameterTuning:
    def __init__(self, A, b, optimizer="GD"):

        self.A = A
        self.b = b

        self.lr_upper  = 1 / A.shape[0]
        self.lr_downer = 1 / (A.shape[0] * 100)

        self.result = {}
        self.optimizer = optimizer

        if self.optimizer == "GD":
            self.learning_rate = np.linspace(self.lr_downer, self.lr_upper)

        elif self.optimizer == "momentum":
            self.learning_rate = np.linspace(self.lr_downer, self.lr_upper)
            self.beta = np.linspace(0,1)

        elif self.optimizer == "Nesterov":
            self.learning_rate = np.linspace(self.lr_downer, self.lr_upper)
            self.beta = np.linspace(0,1)

        else:
            pass

        self.parameter = list()

    def run(self):
        if self.optimizer == "GD":
            self.tune_single_parameter()

        elif self.optimizer == "momentum":
            self.tune_double_parameter()

        elif self.optimizer == "Nesterov":
            self.tune_double_parameter()

        else:
            return NotImplementedError()

        self.visualize()
        self.unpacking_key()

    def tune_single_parameter(self):

        for learning_rate in self.learning_rate:
            key = f"learning_rate: {learning_rate}"
            
            lstsq = LeastSquare(self.A, self.b, learning_rate, optimize_method=self.optimizer)
            lstsq.solve()

            self.result[key] = lstsq.error

    def tune_double_parameter(self):

        for learning_rate in self.learning_rate:
            for beta in self.beta:
                key = f"learning_rate: {learning_rate} beta: {beta}"

                lstsq = LeastSquare(self.A, self.b, learning_rate, optimize_method=self.optimizer)
                lstsq.shared.beta = beta
                lstsq.solve()

                self.result[key] = lstsq.error

    def visualize(self):
        self.keys       = list()
        self.error_list = list()

        k = 0

        for key, value in sorted(self.result.items(), key=lambda x : x[1][-1]):
            if k < 5:
                self.keys.append(key)
                self.error_list.append(self.result[key])

            else:
                break

            k += 1

        plt.figure(figsize=(15,3))

        ## best 5
        for i in range(5):
            plt.subplot(1,5,i+1)
            plt.plot(self.error_list[i])
            plt.title(f"Best-{i+1}")
            plt.loglog(basex=10,basey=10)
            plt.xlim(1,10)
            plt.ylim(1,1000)

        plt.show()

        print(self.keys[0])

    def cut_nan_in_single_parameter(self):
        for learning_rate in self.learning_rate:
            key = f"learning_rate: {learning_rate}"

            value = self.result[key]

            if value[-1] == value[-1]:
                pass

            else:
                del self.result[key]

    def cut_nan_in_double_parameter(self):

        ## cut nan values...
        for learning_rate in self.learning_rate:
            for beta in self.beta:
                key = f"learning_rate: {learning_rate} beta: {beta}"

                value = self.result[key]
                
                if value[-1] == value[-1]:
                    pass

                else:
                    del self.result[key]

    def unpacking_key(self):
        best_key = self.keys[0]

        for string in best_key.split():
            try:
                self.parameter.append(float(string))

            except:
                pass