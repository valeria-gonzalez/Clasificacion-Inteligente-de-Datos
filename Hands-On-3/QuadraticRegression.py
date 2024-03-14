import numpy as np
from RegressionMatrix import RegressionMatrix
from math import sqrt

class QuadraticRegression:
    """Quadratic regression predictive analysis model."""
    def __init__(self, x_train: list, y_train: list)-> None:
        self.x_train = x_train
        self.y_train = y_train
        self.a = None
        self.bx = None
        self.cx2 = None
        
    def fit(self)-> tuple:
        """Fit the quadratic regression model to the dataset.
        The quadratic equation has the form: \n
        y = a + bx + cx^2 where a != 0
        
        Returns: 
            tuple (float, float, float): coefficients (a, bx, cx^2) of the quadratic equation.
        """
        self.a, self.bx, self.cx2 = self.computeQuadraticEquation()
        
        return self.a, self.bx, self.cx2
    
    def predict(self, x_test: list)-> list:
        """Predict the target value for the given features
        
        Args: 
            x_test (list): list of features
        
        Returns: 
            list: predicted target values
        """
        y_pred = [(self.a + (self.bx * x) + (self.cx2 * (x**2))) for x in x_test]
        
        return y_pred
        
    def computeQuadraticEquation(self)-> None:
        """Solves a system of three linear equations using Cramer's rule to find
        the coefficients of a quadratic regression equation.\n
        The quadratic equation has the form: \n
        y = a + bx + cx^2 where a != 0
        """
        matrix = RegressionMatrix(self.x_train, self.y_train, 2)
        regression_matrix = matrix.makeRegressionMatrix()
        transposed_matrix = matrix.transpose()
        
        transposed_dot_matrix = matrix.multiply(transposed_matrix, 
                                                regression_matrix)
        
        inverse_matrix = matrix.inverse(transposed_dot_matrix)
        
        inverse_dot_transposed = matrix.multiply(inverse_matrix, 
                                                 transposed_matrix)
        
        final_result = matrix.multiply(inverse_dot_transposed,
                                       np.array(self.y_train))
        return final_result
    
    def correlationCoefficient(self)-> tuple:
        """Compute the correlation coefficient of the linear regression model.
        
        Returns:
            tuple(float, float): correlation and determination coefficient.
        """
        x_mean = sum(self.x_train) / len(self.x_train)
        y_mean = sum(self.y_train) / len(self.y_train)
        numerator = sum([((x - x_mean) * (y - y_mean)) for x, y in zip(self.x_train, self.y_train)])
        one = sqrt(sum([((x - x_mean) ** 2) for x in self.x_train]))
        two = sqrt(sum([((y - y_mean) ** 2) for y in self.y_train]))
        denominator = one * two
        
        correlation = numerator / denominator
        determination = correlation ** 2
        
        return correlation, determination