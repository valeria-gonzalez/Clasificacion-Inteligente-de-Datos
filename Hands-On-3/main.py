from QuadraticRegression import QuadraticRegression
from LinearRegression import LinearRegression
from CubicRegression import CubicRegression
import pandas as pd

def main():
    file = 'machine_efficiency.csv'
    df = pd.read_csv(file)
    x_name = 'Batch Size'
    y_name = 'Machine Efficiency'
    x_train = df[x_name].tolist()
    y_train = df[y_name].tolist()
    
    linear_regression(x_train, y_train, x_name, y_name)
    quadratic_regression(x_train, y_train, x_name, y_name)
    cubic_regression(x_train, y_train, x_name, y_name)
    
def linear_regression(x_train, y_train, x_name, y_name):
    """ Perform linear regression analysis."""
    linear_model = LinearRegression(x_train, y_train)
    a, bx = linear_model.fit()
    
    known_values = [108, 115, 106]
    known_expected = [95, 96, 95]
    known_predict = linear_model.predict(known_values)
    
    unknown_values = [10, 20, 30]
    unknown_predict = linear_model.predict(unknown_values)
    
    correlation, determination = linear_model.correlationCoefficient()
    
    print(f"""
    ----------------------------------------
    -----------------------------------------
    LINEAR REGRESSION EQUATION ( y = a + bx ):
    -----------------------------------------
    
    {y_name} = {round(a, 5)} + {round(bx, 5)} {x_name} 
    
    ----------------------------------------
    Predicted values for known batch sizes:
    ---------------------------------------
    Known {x_name} (X Values):
    {known_values}
    
    Expected {y_name} (Y Values):
    {known_expected}
    
    Predicted {y_name} (Y Values):
    {known_predict}
    
    ----------------------------------------
    Predicted values for unknown batch sizes:
    ----------------------------------------
    Unknown {x_name} (X Values):
    {unknown_values}
    
    Predicted {y_name} (Y Values):
    {unknown_predict}
    
    ----------------------------------------
    Correlation and Determination Coefficients:
    ----------------------------------------
    Correlation: {round(correlation, 5)}
    
    Determination: {round(determination, 5)}
    """)
    
def quadratic_regression(x_train, y_train, x_name, y_name):
    """ Perform quadratic regression analysis."""
    quadratic_model = QuadraticRegression(x_train, y_train)
    
    a, bx, cx2 = quadratic_model.fit()
    
    known_values = [108, 115, 106]
    known_expected = [95, 96, 95]
    known_predict = quadratic_model.predict(known_values)
    
    unknown_values = [10, 20, 30]
    unknown_predict = quadratic_model.predict(unknown_values)
    
    correlation, determination = quadratic_model.correlationCoefficient()
    
    print(f"""
    ---------------------------------------------------
    ---------------------------------------------------
    QUADRATIC REGRESSION EQUATION ( y = a + bx + cx^2 ):
    ---------------------------------------------------
    
    {y_name} = {round(a, 5)} + {round(bx, 5)} {x_name} + {round(cx2, 5)} {x_name}^2
    
    ----------------------------------------
    Predicted values for known batch sizes:
    ---------------------------------------
    Known {x_name} (X Values):
    {known_values}
    
    Expected {y_name} (Y Values):
    {known_expected}
    
    Predicted {y_name} (Y Values):
    {known_predict}
    
    ----------------------------------------
    Predicted values for unknown batch sizes:
    ----------------------------------------
    Unknown {x_name} (X Values):  
    {unknown_values}
    
    Predicted {y_name} (Y Values):
    {unknown_predict}
    
    ----------------------------------------
    Correlation and Determination Coefficients:
    ----------------------------------------
    Correlation: {round(correlation, 5)}
    
    Determination: {round(determination, 5)}
    """)
    
def cubic_regression(x_train, y_train, x_name, y_name):
    """ Perform cubic regression analysis. """
    cubic_model = CubicRegression(x_train, y_train)
    a, bx, cx2, dx3 = cubic_model.fit()
    
    known_values = [108, 115, 106]
    known_expected = [95, 96, 95]
    known_predict = cubic_model.predict(known_values)
    
    unknown_values = [10, 20, 30]
    unknown_predict = cubic_model.predict(unknown_values)
    
    correlation, determination = cubic_model.correlationCoefficient()
    
    print(f"""
    ---------------------------------------------------------
    ---------------------------------------------------------
    CUBIC REGRESSION EQUATION ( y = a + bx + cx^2 + dx^3 ):
    --------------------------------------------------------
    
    {y_name} = {round(a, 5)} + {round(bx, 5)} {x_name} + {round(cx2, 5)} {x_name}^2 + {round(dx3, 5)} {x_name}^3  
    
    ----------------------------------------
    Predicted values for known batch sizes:
    ---------------------------------------
    Known {x_name} (X Values):
    {known_values}
    
    Expected {y_name} (Y Values):
    {known_expected}
    
    Predicted {y_name} (Y Values):
    {known_predict}
    
    ----------------------------------------
    Predicted values for unknown batch sizes:
    ----------------------------------------
    Unknown {x_name} (X Values):
    {unknown_values}
    
    Predicted {y_name} (Y Values):
    {unknown_predict}
    
    ----------------------------------------
    Correlation and Determination Coefficients:
    ----------------------------------------
    Correlation: {round(correlation, 5)}
    
    Determination: {round(determination, 5)}  
    """)
    
if __name__ == "__main__":
    main()