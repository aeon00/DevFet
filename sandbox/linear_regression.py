import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def train_linear_regression(df):
    # Assuming the last column is the target variable
    X = df.iloc[:, :-1]  # Features (first 4 columns)
    y = df.iloc[:, -1]   # Target (last column)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Print model coefficients and metrics
    print("\nModel Results:")
    print("Feature Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")
    print(f"\nIntercept: {model.intercept_:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Root Mean Square Error: {rmse:.4f}")
    
    return model, scaler

# Example usage:
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'feature4': np.random.normal(0, 1, n_samples),
    }
    
    # Create target variable with some relationship to features
    target = (2 * data['feature1'] + 
             0.5 * data['feature2'] - 
             1.5 * data['feature3'] + 
             3 * data['feature4'] + 
             np.random.normal(0, 0.1, n_samples))
    
    data['target'] = target
    df = pd.DataFrame(data)
    
    # Train the model
    model, scaler = train_linear_regression(df)