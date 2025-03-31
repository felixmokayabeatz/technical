# main.py
from generate_data import generate_churn_data
from data_preprocessing import load_and_explore_data, preprocess_data
from model_training import train_and_evaluate_models, make_predictions
import pandas as pd
import joblib

def main():
    print("Predictive Analytics Using Machine Learning")
    print("Technical Assignment II\n")
    
    # Generate synthetic data
    print("Generating synthetic customer churn data...")
    df = generate_churn_data(n_samples=1000)
    
    # Data exploration
    print("\nExploring the dataset...")
    df = load_and_explore_data('customer_churn_data.csv')
    
    # Data preprocessing
    print("\nPreprocessing the data...")
    X_processed, y, feature_names, preprocessor = preprocess_data(df)
    
    # Model training and evaluation
    print("\nTraining and evaluating models...")
    best_model, X_test, y_test = train_and_evaluate_models(X_processed, y, feature_names)
    
    # Save preprocessor for future use
    joblib.dump(preprocessor, 'preprocessor.pkl')
    
    # Example of using the model for predictions
    print("\nExample of model usage for new data:")
    
    # Create sample new data
    new_data = pd.DataFrame({
        'tenure_months': [20],
        'monthly_charges': [85.50],
        'total_charges': [1710.0],
        'monthly_gb_download': [350.5],
        'gender': ['Male'],
        'senior_citizen': [0],
        'partner': ['Yes'],
        'dependents': ['No'],
        'phone_service': ['Yes'],
        'internet_service': ['Fiber optic'],
        'contract': ['Month-to-month'],
        'payment_method': ['Electronic check'],
        'online_security': ['No']
    })
    
    # Make prediction
    preprocessor = joblib.load('preprocessor.pkl')
    model = joblib.load('best_model.pkl')
    
    prediction = make_predictions(model, new_data, preprocessor)
    print(f"Prediction for new customer: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    
    print("\nProject completed successfully!")
    print("Check the current directory for saved model, visualizations, and output files.")

if __name__ == "__main__":
    main()