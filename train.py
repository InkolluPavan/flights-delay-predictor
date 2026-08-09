from preprocess import load_and_preprocess_data
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

def main():
    print("1. Loading and preprocessing data from CSV...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data('flights.csv')
    
    # 2. Define the candidate models to test
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0, random_state=42)
    }
    
    best_model_name = None
    best_model = None
    best_rmse = float('inf')
    best_r2 = -float('inf')
    
    print("\n2. Training and evaluating models...")
    print("-" * 50)
    
    # 3. Iterate through models, train, and evaluate
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f" > {name} Results -> RMSE: {rmse:.4f} | R2 Score: {r2:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_model_name = name
            best_model = model
            
    print("-" * 50)
    print(f"\n🏆 Best Model Selected: {best_model_name}")
    print(f"   - RMSE: {best_rmse:.4f}")
    print(f"   - R2 Score: {best_r2:.4f}")
    
    # 4. Bundle model and training columns together for the GUI
    package = {
        'model': best_model,
        'expected_columns': X_train.columns.tolist()
    }
    
    model_filename = 'flight_delay_model.pkl'
    joblib.dump(package, model_filename)
    print(f"\nWinning model and metadata successfully saved as '{model_filename}'!")

if __name__ == "__main__":
    main()