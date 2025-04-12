import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from utils import BRANCH_CODES, FIELD_OPTIONS, get_detailed_station_info

def predict_ps_stations(user_profile, model, feature_columns, ps_stations, historical_data, top_n=10):
    """
    Predict suitable PS stations based on the user's profile.
    
    Args:
        user_profile: Dictionary containing user profile information
        model: Trained machine learning model
        feature_columns: List of feature column names used in training
        ps_stations: List of unique PS stations
        historical_data: DataFrame with historical allocation data
        top_n: Number of recommendations to return
        
    Returns:
        list: List of dictionaries containing recommended PS stations with details
    """
    if model is None or not feature_columns or not ps_stations:
        return []
    
    # Create a DataFrame for prediction
    user_df = pd.DataFrame([user_profile])
    
    # Prepare features for prediction
    X_pred = prepare_features_for_prediction(user_df, model, feature_columns)
    
    if X_pred is None:
        return []
    
    # Get probability predictions for all stations
    try:
        probabilities = model.predict_proba(X_pred)[0]
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return []
    
    # Get indices of top N stations by probability
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    
    # Create recommendations
    recommendations = []
    for idx in top_indices:
        station = ps_stations[idx]
        probability = probabilities[idx]
        
        # Get additional station details
        station_details = get_detailed_station_info(station, historical_data, user_profile['Branch'])
        
        recommendation = {
            'station': station,
            'score': f"{probability:.4f}",
            **station_details
        }
        
        recommendations.append(recommendation)
    
    return recommendations


def prepare_features_for_prediction(user_df, model, feature_columns):
    """
    Prepare user input for prediction by applying the same transformations used during training.
    
    Args:
        user_df: DataFrame with user input
        model: Trained model with encoders
        feature_columns: List of feature column names used in training
        
    Returns:
        DataFrame: Processed features ready for prediction
    """
    processed_df = user_df.copy()
    
    # Process each feature using the encoders from training
    for feature in model.original_features:
        if feature not in processed_df.columns:
            print(f"Warning: Feature '{feature}' not found in user input.")
            return None
        
        # For categorical features, apply the same encoding
        if feature in model.encoders:
            # Get the encoder
            encoder = model.encoders[feature]
            
            # Transform the feature
            try:
                # Handle new categories not seen during training
                if processed_df[feature].iloc[0] not in encoder.classes_:
                    # Use the most frequent category instead
                    processed_df[feature] = encoder.classes_[0]
                
                processed_df[f"{feature}_encoded"] = encoder.transform(processed_df[feature].astype(str))
            except Exception as e:
                print(f"Error encoding feature '{feature}': {str(e)}")
                # Set to the first class as fallback
                processed_df[f"{feature}_encoded"] = 0
    
    # Select only the encoded features needed for prediction
    X_pred = processed_df[[col for col in model.feature_columns if col in processed_df.columns]]
    
    # Ensure all required columns are present
    missing_cols = [col for col in model.feature_columns if col not in X_pred.columns]
    if missing_cols:
        print(f"Warning: Missing columns for prediction: {missing_cols}")
        
        # Add missing columns with default values (0)
        for col in missing_cols:
            X_pred[col] = 0
    
    # Ensure column order matches training data
    X_pred = X_pred[model.feature_columns]
    
    return X_pred


def get_surprise_recommendation(user_profile, model, feature_columns, ps_stations, historical_data):
    """
    Generate a surprise recommendation that's different from the top predicted stations
    but still potentially suitable.
    
    Args:
        user_profile: Dictionary containing user profile information
        model: Trained machine learning model
        feature_columns: List of feature column names used in training
        ps_stations: List of unique PS stations
        historical_data: DataFrame with historical allocation data
        
    Returns:
        dict: Dictionary containing surprise recommendation with details
    """
    if model is None or not feature_columns or not ps_stations:
        return None
    
    # Get standard predictions first
    standard_predictions = predict_ps_stations(
        user_profile, model, feature_columns, ps_stations, historical_data, top_n=5
    )
    
    if not standard_predictions:
        return None
    
    # Extract the standard recommendation stations
    standard_stations = [pred['station'] for pred in standard_predictions]
    
    # Create a DataFrame for prediction
    user_df = pd.DataFrame([user_profile])
    
    # Prepare features for prediction
    X_pred = prepare_features_for_prediction(user_df, model, feature_columns)
    
    if X_pred is None:
        return None
    
    # Get probability predictions for all stations
    try:
        probabilities = model.predict_proba(X_pred)[0]
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None
    
    # Get stations with non-zero probability but exclude top recommendations
    potential_stations = []
    for idx, prob in enumerate(probabilities):
        station = ps_stations[idx]
        if prob > 0.01 and station not in standard_stations:
            potential_stations.append((station, prob))
    
    # If no suitable stations found, try a completely random station
    if not potential_stations:
        # Exclude standard stations
        surprise_candidates = [st for st in ps_stations if st not in standard_stations]
        
        if not surprise_candidates:
            # If somehow all stations are in standard recommendations, just pick a random one
            surprise_candidates = ps_stations
        
        # Randomly select a station
        surprise_station = random.choice(surprise_candidates)
        
        # Get the probability
        surprise_prob = probabilities[ps_stations.index(surprise_station)]
    else:
        # Weight the selection by probability
        weights = [prob for _, prob in potential_stations]
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        
        # Randomly select a station based on weights
        selected_idx = np.random.choice(len(potential_stations), p=normalized_weights)
        surprise_station, surprise_prob = potential_stations[selected_idx]
    
    # Get additional station details
    station_details = get_detailed_station_info(surprise_station, historical_data, user_profile['Branch'])
    
    # Create a reason for the surprise recommendation
    reason = generate_surprise_reason(
        surprise_station, 
        user_profile, 
        historical_data,
        surprise_prob
    )
    
    # Create historical trend data
    historical_trend = None
    if 'Year' in historical_data.columns:
        station_data = historical_data[historical_data['PS_Station'] == surprise_station]
        if not station_data.empty:
            trend = station_data.groupby('Year').size()
            if len(trend) > 1:
                historical_trend = trend
    
    # Create the surprise recommendation
    surprise_recommendation = {
        'station': surprise_station,
        'score': f"{surprise_prob:.4f}",
        'reason': reason,
        'historical_trend': historical_trend,
        **station_details
    }
    
    return surprise_recommendation


def generate_surprise_reason(station, user_profile, historical_data, probability):
    """
    Generate a personalized reason why this surprise recommendation might be interesting.
    
    Args:
        station: The recommended PS station
        user_profile: Dictionary containing user profile information
        historical_data: DataFrame with historical allocation data
        probability: The prediction probability
        
    Returns:
        str: A personalized reason for the recommendation
    """
    # Filter historical data for this station
    station_data = historical_data[historical_data['PS_Station'] == station]
    
    reasons = []
    
    # Check if students from the same branch have been allocated here
    if 'Branch' in station_data.columns and not station_data.empty:
        branch_matches = station_data[station_data['Branch'] == user_profile['Branch']]
        
        if not branch_matches.empty:
            branch_count = len(branch_matches)
            total_count = len(station_data)
            percentage = (branch_count / total_count) * 100
            
            if percentage > 0:
                reasons.append(f"{percentage:.1f}% of students at this station are from your branch ({user_profile['Branch']}).")
    
    # Check CGPA distribution
    if 'CGPA' in station_data.columns and not station_data.empty:
        avg_cgpa = station_data['CGPA'].mean()
        min_cgpa = station_data['CGPA'].min()
        user_cgpa = user_profile['CGPA']
        
        if user_cgpa >= avg_cgpa:
            reasons.append(f"Your CGPA ({user_cgpa}) is higher than the average CGPA ({avg_cgpa:.2f}) of students at this station.")
        elif user_cgpa >= min_cgpa:
            reasons.append(f"Your CGPA ({user_cgpa}) is within the range of students who have been placed at this station (minimum {min_cgpa:.2f}).")
    
    # Add a comment about the probability
    if probability < 0.1:
        reasons.append("This is an unconventional choice, which might offer unique experiences not typically associated with your profile.")
    elif probability < 0.3:
        reasons.append("While not the most common choice for your profile, this station offers interesting opportunities worth considering.")
    else:
        reasons.append("This station has significant potential alignment with aspects of your profile, though it wasn't among the top recommendations.")
    
    # Add a general reason if no specific reasons were found
    if not reasons:
        reasons.append("This station offers a different experience that might broaden your professional horizons beyond the typical choices for your profile.")
    
    # Combine reasons into a paragraph
    return " ".join(reasons)
