from RegressionMatrix import RegressionMatrix
import numpy as np
from math import sqrt

class CubicRegression():
    """Cubic regression predictive analysis model."""
    def __init__(self, x_train: list, y_train: list)-> None:
        self.x_train = x_train
        self.y_train = y_train
        self.a = None
        self.bx = None
        self.cx2 = None
        self.dx3 = None
        
    def fit(self)-> tuple:
        """Fit the cubic regression model to the dataset.
        The cubic equation has the form: \n
        y = a + bx + cx^2 + dx^3
        
        Returns: 
            tuple (float, float, float, float): coefficients (a, bx, cx^2, dx^3) of the cubic equation.
        """
        self.a, self.bx, self.cx2, self.dx3 = self.computeCubicEquation()
        
        return self.a, self.bx, self.cx2, self.dx3
    
    def predict(self, x_test: list)-> list:
        """Predict the target value for the given features
        
        Args: 
            x_test (list): list of features
        
        Returns: 
            list: predicted target values
        """
        y_pred = [(self.a + (self.bx * x) + (self.cx2 * (x**2))+ (self.dx3 * (x**3))) for x in x_test]
        
        return y_pred
    
    def computeCubicEquation(self)-> np.array:
        """Compute the coefficients for the linear regression equation.\n
        The linear equation has the form: \n
        y = a + bx + cx^2 + dx^3

        Returns:
            np.array: coefficients (a, bx, cx^2, dx^3) of the cubic equation.
        """
        matrix = RegressionMatrix(self.x_train, self.y_train, 3)
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
        if(self.a == None): 
            self.fit()
            
        y_predict = self.predict(self.x_train)
        y_mean = sum(self.y_train) / len(self.y_train)
        
        numerator = sum([((y_pred - y_mean) ** 2) for y_pred in y_predict])
        denominator = sum([((y - y_mean) ** 2) for y in self.y_train])
        determination = numerator / denominator
        
        correlation = sqrt(determination)
        
        return correlation, determination
        
        
        