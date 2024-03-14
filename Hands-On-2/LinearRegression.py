from math import sqrt

class LinearRegression:
    """Linear regression predictive analysis model."""
    
    def __init__(self, x_train: list, y_train: list) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_sum = None
        self.y_sum = None
        self.xy_sum = None
        self.x2_sum = None
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
    
    def computeSums(self)-> None:
        """Compute the sums needed for the linear regression model:
        - Sum of x values
        - Sum of y values
        - Sum of x*y values
        - Sum of x^2 values
        """
        self.x_sum = sum(self.x_train)
        self.y_sum = sum(self.y_train)
        self.xy_sum = sum([x*y for x, y in zip(self.x_train, self.y_train)])
        self.x2_sum = sum([x**2 for x in self.x_train])
    
    def computeLinearEquation(self)-> tuple:
        """Compute the coefficients for the linear regression equation.\n
        The linear equation has the form: \n
        y = a + bx
        
        Returns:
            tuple (float, float): coefficients (a, bx) of the linear equation.
        """
        if self.x_sum == None:
            self.computeSums()
            
        x_train_size = len(self.x_train)
        
        a = ((self.y_sum * self.x2_sum) - (self.x_sum * self.xy_sum)) 
        a /= ((x_train_size * self.x2_sum) - (self.x_sum ** 2))
        
        bx = ((x_train_size * self.xy_sum) - (self.x_sum * self.y_sum)) 
        bx /= ((x_train_size * self.x2_sum) - (self.x_sum ** 2))
        
        a = round(a)
        bx = round(bx)
        return a, bx
        
    def computeCorrelationCoefficient(self)-> tuple:
        """Compute the correlation and determination coefficients.
        
        Returns:
            tuple (float, float): Correlation and determination coefficients, respectively
        """
        if self.x_sum == None:
            self.computeSums()
            
        x_train_size = len(self.x_train)
        y2_sum = sum([y**2 for y in self.y_train])
        
        numerator = ((x_train_size * self.xy_sum) - (self.x_sum * self.y_sum)) 
        denominator = (sqrt((x_train_size * self.x2_sum - (self.x_sum ** 2)) * (x_train_size * y2_sum - (self.y_sum ** 2))))
        
        correlation = numerator / denominator
        determination = correlation ** 2
        
        return correlation, determination