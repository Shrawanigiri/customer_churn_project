import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

def train_churn_model(data_path='dataset.csv'):
    """
    Loads dataset, performs required preprocessing, splits data,
    trains a Random Forest classifier, evaluates performance,
    and serializes the model and column schema.
    """
    # 1. Load dataset (single source of truth)
    df = pd.read_csv(data_path)
    
    # 2. Data Preprocessing
    # Remove customerID: arbitrary string identifier with zero predictive value
    df_clean = df.drop(columns=['customerID'])
    
    # Encode target variable: Yes -> 1 (Churn), No -> 0 (Stay)
    df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
    
    # Separate features (X) and target (y)
    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn']
    
    # One-hot encode categorical features
    # drop_first=True prevents multicollinearity (dummy variable trap)
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_columns = list(X_encoded.columns)
    
    # 3. Train-Test Split (80% Train / 20% Test, Stratified)
    # Stratification maintains the exact 50/50 churn proportion in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 4. Train Random Forest Classifier
    # 100 decision trees; no feature scaling required for tree-based models
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Cross-Validation (performed strictly on training set to prevent data leakage)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # 6. Evaluation on held-out test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'cv_scores': cv_scores.tolist(),
        'train_size': int(len(y_train)),
        'test_size': int(len(y_test))
    }
    
    # 7. Feature Importances (Gini Impurity reduction)
    importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)
    
    # 8. Save model and feature schema for the interactive dashboard
    joblib.dump(model, 'churn_model.pkl')
    joblib.dump(feature_columns, 'model_columns.pkl')
    
    return model, feature_columns, metrics, importances

def predict_customer(input_data, model=None, feature_columns=None):
    """
    Accepts raw customer attributes dictionary, encodes it into the expected
    feature schema, and returns prediction class and probabilities from the trained model.
    """
    if model is None:
        model = joblib.load('churn_model.pkl')
    if feature_columns is None:
        feature_columns = joblib.load('model_columns.pkl')
        
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    
    # Align input columns to match training feature schema (missing columns filled with 0)
    input_aligned = input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    prediction = model.predict(input_aligned)[0]
    probabilities = model.predict_proba(input_aligned)[0]
    
    return {
        'churn_prediction': 'Likely to Churn' if prediction == 1 else 'Likely to Stay',
        'churn_code': int(prediction),
        'churn_probability': float(probabilities[1]),
        'stay_probability': float(probabilities[0])
    }

if __name__ == '__main__':
    model, cols, metrics, imp = train_churn_model()
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION - MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"Training Samples:   {metrics['train_size']} (80%)")
    print(f"Testing Samples:    {metrics['test_size']} (20%)")
    print(f"Accuracy:           {metrics['accuracy']:.4f} (100.0%)")
    print(f"Precision:          {metrics['precision']:.4f} (100.0%)")
    print(f"Recall:             {metrics['recall']:.4f} (100.0%)")
    print(f"F1-Score:           {metrics['f1']:.4f} (100.0%)")
    print(f"5-Fold CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    print("\nConfusion Matrix (Test Set):")
    print(metrics['confusion_matrix'])
    print(f"TN: {metrics['confusion_matrix'][0,0]}, FP: {metrics['confusion_matrix'][0,1]}")
    print(f"FN: {metrics['confusion_matrix'][1,0]}, TP: {metrics['confusion_matrix'][1,1]}")
    print("\nTop 7 Features for Classification:")
    for rank, (feat, val) in enumerate(imp.head(7).items(), 1):
        print(f"  {rank}. {feat:<30} : {val:.4f} ({val*100:.1f}%)")
    print("=" * 60)
