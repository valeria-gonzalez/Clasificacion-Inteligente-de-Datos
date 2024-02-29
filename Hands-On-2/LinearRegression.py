from math import sqrt

class LinearRegression:
    def __init__(self) -> None:
        self.beta0 = 0
        self.beta1 = 0
    
    def fit(self, x_train: list, y_train: list) -> tuple: 
        """Fit the model to the dataset
        Args: 
            x_train (list): list of features
            y_train (list): list of target values
        Returns: 
            tuple of beta values
        """
        x_sum, y_sum, xy_sum, x2_sum = self.compute_sums(x_train, y_train)
        self.beta0, self.beta1 = self.compute_betas(
            x_sum, 
            y_sum, 
            xy_sum, 
            x2_sum, 
            len(x_train)
        )
        
        return self.beta0, self.beta1
    
    def predict(self, x_test: list) -> list:
        """Predict the target value for the given features
        Args: 
            x_test (list): list of features
        Returns: 
            list of predicted target values
        """
        y_pred = [self.beta0 + (self.beta1 * x) for x in x_test]
        
        return y_pred
    
    def compute_sums(self, x_train: list, y_train: list) -> tuple:
        """Compute the sums needed for the linear regression model:
        sum of x values, sum of y values, sum of x*y values, sum of x^2 values
        Args: 
            x_train (list): list of features
            y_train (list): list of target values
        Returns: 
            tuple of sums 
        """
        x_sum = sum(x_train)
        y_sum = sum(y_train)
        xy_sum = sum([x*y for x, y in zip(x_train, y_train)])
        x2_sum = sum([x**2 for x in x_train])
        
        return x_sum, y_sum, xy_sum, x2_sum
    
    def compute_betas(
        self, x_sum: float, y_sum: float, xy_sum: float, x2_sum: float, n: int
    ) -> tuple:
        """Compute the beta0 and beta1 values for the linear regression model

        Args:
            x_sum (int): sum of x values
            y_sum (int): sum of y values
            xy_sum (int): sum of x*y values
            x2_sum (int): sum of x^2 values
        Returns: 
            tuple of beta values
        """
        beta0 = ((y_sum * x2_sum) - (x_sum * xy_sum)) 
        beta0 /= ((n * x2_sum) - (x_sum ** 2))
        
        beta1 = ((n * xy_sum) - (x_sum * y_sum)) 
        beta1 /= ((n * x2_sum) - (x_sum ** 2))
        
        return round(beta0), round(beta1)
    
    def compute_correlation_deter_coeff(
        self, x_train: list, y_train: list
    ) -> tuple:
        """Compute the correlation and determination coefficients.

        Args:
            x_train (list): list of features
            y_train (list): list of target values
        Returns:
            tuple of correlation and determination coefficients
        """
        n = len(x_train)
        x_sum, y_sum, xy_sum, x2_sum = self.compute_sums(x_train, y_train)
        y2_sum = sum([y**2 for y in y_train])
        
        correlation = ((n * xy_sum) - (x_sum * y_sum)) 
        correlation /= (sqrt((n * x2_sum - (x_sum ** 2)) * (n * y2_sum - (y_sum ** 2))))
        
        determination = correlation ** 2
        
        return correlation, determination
    
    