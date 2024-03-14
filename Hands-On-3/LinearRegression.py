from RegressionMatrix import RegressionMatrix
import numpy as np
from math import sqrt

class LinearRegression:
    """Linear regression predictive analysis model."""
    
    def __init__(self, x_train: list, y_train: list) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.a = None
        self.bx = None
        
    def fit(self)-> tuple: 
        """Fit the linear regression model to the dataset.
        The linear equation has the form: \n
        y = a + bx
        
        Returns: 
            tuple (float, float): coefficients (a, bx) of the linear equation.
        """
        self.a, self.bx = self.computeLinearEquation()
        
        return self.a, self.bx
    
    def predict(self, x_test: list)-> list:
        """Predict the target value for the given features
        
        Args: 
            x_test (list): list of features
        
        Returns: 
            list: predicted target values
        """
        y_pred = [self.a + (self.bx * x) for x in x_test]
        
        return y_pred
    
    def computeLinearEquation(self)-> np.array:
        """Compute the coefficients for the linear regression equation.\n
        The linear equation has the form: \n
        y = a + bx
        
        Returns:
            np.array: coefficients (a, bx) of the linear equation.
        """
        matrix = RegressionMatrix(self.x_train, self.y_train, 1)
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
        
        
        
