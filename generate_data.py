
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def generate_churn_data(n_samples=1000, random_state=42):
    """Generate synthetic customer churn data."""
    
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=random_state
    )
    
    
    df = pd.DataFrame(X, columns=[
        'tenure_months',
        'monthly_charges',
        'total_charges',
        'internet_service_type_score',
        'online_security_score',
        'tech_support_score',
        'streaming_tv_score',
        'contract_length_score',
        'payment_method_score',
        'monthly_gb_download'
    ])
    
    
    df['tenure_months'] = (df['tenure_months'] * 20 + 30).astype(int)
    df['monthly_charges'] = (df['monthly_charges'] * 50 + 70).round(2)
    df['total_charges'] = (df['total_charges'] * 1000 + 2000).round(2)
    df['monthly_gb_download'] = (df['monthly_gb_download'] * 500 + 200).round(2)
    
    
    df['gender'] = np.random.choice(['Male', 'Female'], size=n_samples)
    df['senior_citizen'] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    df['partner'] = np.random.choice(['Yes', 'No'], size=n_samples)
    df['dependents'] = np.random.choice(['Yes', 'No'], size=n_samples)
    df['phone_service'] = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.9, 0.1])
    
    
    service_options = ['DSL', 'Fiber optic', 'None']
    df['internet_service'] = pd.qcut(
        df['internet_service_type_score'], 
        q=[0, 0.3, 0.7, 1.0], 
        labels=service_options
    )
    
    contract_options = ['Month-to-month', 'One year', 'Two year']
    df['contract'] = pd.qcut(
        df['contract_length_score'], 
        q=[0, 0.5, 0.8, 1.0], 
        labels=contract_options
    )
    
    payment_options = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
    df['payment_method'] = pd.qcut(
        df['payment_method_score'], 
        q=[0, 0.25, 0.5, 0.75, 1.0], 
        labels=payment_options
    )
    
    security_options = ['Yes', 'No', 'No internet service']
    df['online_security'] = pd.qcut(
        df['online_security_score'], 
        q=[0, 0.3, 0.7, 1.0], 
        labels=security_options
    )
    
    
    df['churn'] = y
    df['churn'] = df['churn'].map({0: 'No', 1: 'Yes'})
    
    
    df = df.drop([
        'internet_service_type_score',
        'online_security_score',
        'tech_support_score',
        'streaming_tv_score',
        'contract_length_score',
        'payment_method_score'
    ], axis=1)
    
    
    for col in ['tenure_months', 'monthly_charges', 'online_security', 'payment_method']:
        mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
        df.loc[mask, col] = np.nan
    
    
    df.to_csv('customer_churn_data.csv', index=False)
    print(f"Dataset generated with {n_samples} samples and saved to 'customer_churn_data.csv'")
    
    return df

if __name__ == "__main__":
    generate_churn_data()