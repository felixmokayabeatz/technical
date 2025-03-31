
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

def train_and_evaluate_models(X, y, feature_names):
    """Train multiple models and evaluate their performance."""
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        
        y_pred = model.predict(X_test)
        
        
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        
        
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
        
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
        plt.close()
    
    
    results_df = pd.DataFrame(results).T
    print("\nModel Comparison:")
    print(results_df)
    
    
    plt.figure(figsize=(12, 6))
    results_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    
    best_model_name = results_df['f1'].idxmax()
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name} with F1 score: {results_df.loc[best_model_name, 'f1']:.4f}")
    
    
    print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
    best_tuned_model = tune_hyperparameters(best_model_name, best_model, X_train, y_train, X_test, y_test)
    
    
    if hasattr(best_tuned_model, 'feature_importances_'):
        feature_importance = best_tuned_model.feature_importances_
        
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        
        print("\nTop 10 Important Features:")
        print(importance_df.head(10))
        
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title(f'Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    
    joblib.dump(best_tuned_model, 'best_model.pkl')
    print("Best model saved as 'best_model.pkl'")
    
    return best_tuned_model, X_test, y_test

def tune_hyperparameters(model_name, model, X_train, y_train, X_test, y_test):
    """Tune hyperparameters for the best model."""
    
    param_grid = {}
    
    if model_name == "Logistic Regression":
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['liblinear', 'saga']
        }
    elif model_name == "Decision Tree":
        param_grid = {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_name == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif model_name == "XGBoost":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters: {best_params}")
    
    
    y_pred = best_model.predict(X_test)
    
    print("\nTuned Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    
    y_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    return best_model

def make_predictions(model, X_new, preprocessor=None):
    """Make predictions on new data."""
    
    if preprocessor is not None:
        X_new_processed = preprocessor.transform(X_new)
    else:
        X_new_processed = X_new
    
    predictions = model.predict(X_new_processed)
    
    return predictions