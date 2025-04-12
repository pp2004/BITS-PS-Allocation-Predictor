import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from utils import BRANCH_CODES

def train_model(data, selected_features):
    """
    Train a machine learning model to predict PS station allocations.
    
    Args:
        data: pandas DataFrame with processed data
        selected_features: list of features to use for prediction
        
    Returns:
        trained_model: The trained machine learning model
        feature_columns: List of feature column names used in training
        ps_stations: List of unique PS stations in the training data
    """
    # Make a copy of the data
    df = data.copy()
    
    # Ensure all required features are available
    missing_features = [feat for feat in selected_features if feat not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Filter out missing features
        selected_features = [feat for feat in selected_features if feat in df.columns]
    
    if not selected_features:
        print("Error: No valid features available for training")
        return None, [], []
    
    # Prepare the feature columns
    feature_columns = selected_features.copy()
    
    # Initialize encoders for categorical features
    encoders = {}
    
    # Process each feature
    for feature in feature_columns:
        if df[feature].dtype == 'object' or df[feature].dtype == 'category':
            # Encode categorical features
            encoder = LabelEncoder()
            df[f"{feature}_encoded"] = encoder.fit_transform(df[feature].astype(str))
            encoders[feature] = encoder
            
            # Replace original feature with encoded version in the feature list
            feature_columns[feature_columns.index(feature)] = f"{feature}_encoded"
    
    # Prepare target variable (PS Station)
    target_encoder = LabelEncoder()
    df['PS_Station_encoded'] = target_encoder.fit_transform(df['PS_Station'].astype(str))
    
    # Get the list of unique PS stations
    ps_stations = list(target_encoder.classes_)
    
    # Split data into training and testing sets
    X = df[feature_columns]
    y = df['PS_Station_encoded']
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except ValueError as e:
        print(f"Error splitting data: {str(e)}")
        # If splitting fails, use all data for training
        X_train, y_train = X, y
        X_test, y_test = X, y
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model if we have test data
    if len(X_test) > 0 and len(y_test) > 0:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    # Store encoders with the model for prediction
    model.encoders = encoders
    model.target_encoder = target_encoder
    model.feature_columns = feature_columns
    model.original_features = selected_features
    
    return model, selected_features, ps_stations
