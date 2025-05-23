
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_explore_data(file_path):
    """Load and explore the customer churn dataset."""
    
    
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with shape: {df.shape}")
    
    
    print("\nDataset info:")
    print(df.info())
    
    
    print("\nBasic statistics:")
    print(df.describe())
    
    
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    
    plt.figure(figsize=(8, 5))
    sns.countplot(x='churn', data=df)
    plt.title('Distribution of Churn')
    plt.savefig('churn_distribution.png')
    plt.close()
    
    
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    return df

def preprocess_data(df):
    """Preprocess the dataset for modeling."""
    
    
    print("\nHandling missing values...")
    
    
    processed_df = df.copy()
    
    
    X = processed_df.drop('churn', axis=1)
    y = processed_df['churn'].map({'No': 0, 'Yes': 1})
    
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    
    X_processed = preprocessor.fit_transform(X)
    
    ohe_feature_names = []
    for i, feature in enumerate(categorical_features):
        categories = preprocessor.transformers_[1][1].named_steps['onehot'].categories_[i]
        for category in categories:
            ohe_feature_names.append(f"{feature}_{category}")
    
    feature_names = numeric_features + ohe_feature_names
    
    print(f"Processed data shape: {X_processed.shape}")
    print(f"Number of features after preprocessing: {X_processed.shape[1]}")
    
    return X_processed, y, feature_names, preprocessor

if __name__ == "__main__":
    df = load_and_explore_data('customer_churn_data.csv')
    X_processed, y, feature_names, preprocessor = preprocess_data(df)