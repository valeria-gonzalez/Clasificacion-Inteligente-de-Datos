import numpy as np
import numpy.typing as npt

class RegressionMatrix:
    def __init__(self, x_train: list, y_train: list, order: int)-> None:
        self.x_train = x_train
        self.y_train = y_train
        self.order = order
        self.matrix = self.makeRegressionMatrix()
    
    def makeRegressionMatrix(self)-> np.array:
        """Create a matrix from the given dataset."""
        matrix = []
        for x in self.x_train:
            row = [x**i for i in range(self.order + 1)]
            matrix.append(row)
            
        # print(np.array(matrix))
        return np.array(matrix)
    
    def transpose(self)-> np.array:
        """Transpose the matrix."""
        return np.transpose(self.matrix)
    
    def multiply(self, matrix_a: np.array, matrix_b: np.array)-> np.array:
        """Multiply two matrices."""
        return np.dot(matrix_a, matrix_b)
    
    def inverse(self, matrix: np.array):
        """Find the inverse of a matrix."""
        return np.linalg.inv(matrix)
            
            
            
        
    
            
            
        
        