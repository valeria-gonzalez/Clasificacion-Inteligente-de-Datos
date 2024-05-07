from math import sqrt
import numpy as np

class Maths:
    def __init__(self):
        self.hi = None
    
    def euclideanDistance(self, i: np.array, j: np.array) -> float:
        instance_sum = sum([(xi - xj) ** 2 for xi, xj in zip(i, j)])
        return sqrt(instance_sum)
        