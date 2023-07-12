import random
import math
import matplotlib.pyplot as plt

class RejectionSampling():
    def __init__(self):
        self.proposal = 1 / math.sqrt(math.pi * 2)
        pass
    def gaussian_sampling(self):
        result = self._sampling(self._gaussian)
        return result
    
    def _sampling(self, target):
        x = random.uniform(-1 / self.proposal, 1/ self.proposal)
        ratio = target(x) / self.proposal
        sample = random.uniform(0, 0.999)
        while sample > ratio:
            x = random.uniform(-1 / self.proposal, 1/ self.proposal)
            ratio = target(x) / self.proposal
            sample = random.uniform(0, 0.999)
        return x, sample
    def _gaussian(self, x):
        return math.exp(x**2 / -2) / math.sqrt(2 * math.pi)

    pass

sample = RejectionSampling()

x_value = []
y_value = []
for _ in range(2000):
    
    x, y = sample.gaussian_sampling()
    x_value.append(x)
    y_value.append(y)
plt.plot(x_value, y_value, 'ro')
plt.show()
