from LinearRegression import LinearRegression
import pandas as pd

def main():
    file = 'benetton.csv'
    df = pd.read_csv(file)
    x_train = df['advertising'].tolist()
    y_train = df['sales'].tolist()
    
    model = LinearRegression()
    
    # Calculate linear regression ecuation
    beta0, beta1 = model.fit(x_train, y_train)
    
    print(f"""
    -------------------------
    Regression ecuation:
    -------------------------
    y = {beta0} + {beta1}x1
    """)
    
    # Predict Y for given X
    y_predict = model.predict(x_train)
    
    predictions_df = pd.DataFrame()
    predictions_df['advertising'] = x_train
    predictions_df['sales'] = y_train
    predictions_df['predicted sales'] = y_predict
    
    print(f"""
    ------------------
    Given X predict Y: 
    ------------------
    """)
    print(predictions_df)
    
    # Calculate correlation and determination coefficients
    correlation_coeff, determination_coeff = model.compute_correlation_deter_coeff(x_train, y_train)
    
    print(f"""
    ----------------------------------------
    Correlation and deviation determination: 
    ----------------------------------------
    Correlation coefficient: {correlation_coeff}
    
    Determination coefficient: {determination_coeff}
    """)
    
    # Predict five values that aren't in the dataset
    advertising = [700, 2000, 300, 225, 3000]
    sales_predictions = model.predict(advertising)
    
    predictions_df = pd.DataFrame()
    predictions_df['sales'] = sales_predictions
    predictions_df['advertising'] = advertising
    
    print(f"""
    ------------------
    5 Predictions: 
    ------------------
    """)
    print(predictions_df)
    print()
    

if __name__ == "__main__":
    main()