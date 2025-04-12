import pandas as pd
import numpy as np
import re
from utils import extract_branch_from_id, BRANCH_CODES

def process_excel_data(uploaded_file):
    """
    Process uploaded Excel files containing PS allocation data.
    
    Args:
        uploaded_file: The uploaded Excel file object
        
    Returns:
        pandas.DataFrame: Processed data
    """
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        
        # Check if the dataframe is empty
        if df.empty:
            return None
            
        # Identify and standardize column names
        df_columns = [col.lower() for col in df.columns]
        
        # Map standard column names
        column_mapping = {}
        
        # Student ID column
        for col in df_columns:
            if 'id' in col.lower() or 'number' in col.lower() or 'student' in col.lower():
                column_mapping[df.columns[df_columns.index(col)]] = 'Student_ID'
        
        # CGPA column
        for col in df_columns:
            if 'cgpa' in col.lower() or 'gpa' in col.lower():
                column_mapping[df.columns[df_columns.index(col)]] = 'CGPA'
        
        # PS Station column
        for col in df_columns:
            if 'ps' in col.lower() or 'station' in col.lower() or 'allocation' in col.lower():
                column_mapping[df.columns[df_columns.index(col)]] = 'PS_Station'
        
        # Apply column mapping if found
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Check if required columns exist
        required_columns = ['Student_ID', 'CGPA', 'PS_Station']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try to infer missing columns
            if 'Student_ID' in missing_columns and any('id' in col.lower() for col in df.columns):
                id_col = next(col for col in df.columns if 'id' in col.lower())
                df = df.rename(columns={id_col: 'Student_ID'})
                missing_columns.remove('Student_ID')
            
            if 'CGPA' in missing_columns and any('gpa' in col.lower() for col in df.columns):
                gpa_col = next(col for col in df.columns if 'gpa' in col.lower())
                df = df.rename(columns={gpa_col: 'CGPA'})
                missing_columns.remove('CGPA')
            
            if 'PS_Station' in missing_columns and any('station' in col.lower() for col in df.columns):
                station_col = next(col for col in df.columns if 'station' in col.lower())
                df = df.rename(columns={station_col: 'PS_Station'})
                missing_columns.remove('PS_Station')
                
        if missing_columns:
            # If still missing columns, try to create them if possible
            if 'Student_ID' in missing_columns:
                # If no student ID column, create a sequential one
                df['Student_ID'] = [f"TEMP{i+1}" for i in range(len(df))]
            
            # For other missing columns, we can't proceed
            if 'CGPA' in missing_columns or 'PS_Station' in missing_columns:
                return None
        
        return df
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        return None


def clean_data(df):
    """
    Clean and preprocess the data for analysis and model training.
    
    Args:
        df: pandas DataFrame with raw data
        
    Returns:
        pandas.DataFrame: Cleaned dataframe with derived features
    """
    # Create a copy to avoid modifying the original data
    cleaned_df = df.copy()
    
    # Extract branch from student ID if branch column doesn't exist
    if 'Branch' not in cleaned_df.columns:
        cleaned_df['Branch'] = cleaned_df['Student_ID'].apply(extract_branch_from_id)
    
    # Convert CGPA to float
    if 'CGPA' in cleaned_df.columns:
        cleaned_df['CGPA'] = pd.to_numeric(cleaned_df['CGPA'], errors='coerce')
        
        # Handle missing or invalid CGPA values
        cleaned_df['CGPA'].fillna(cleaned_df['CGPA'].median(), inplace=True)
    
    # Standardize PS station names
    if 'PS_Station' in cleaned_df.columns:
        # Convert to string
        cleaned_df['PS_Station'] = cleaned_df['PS_Station'].astype(str)
        
        # Remove leading/trailing spaces
        cleaned_df['PS_Station'] = cleaned_df['PS_Station'].str.strip()
        
        # Replace common abbreviations or typos
        station_mapping = {
            # Example mappings - would need to be expanded based on actual data
            'ABC Corp.': 'ABC Corporation',
            'ABC corp': 'ABC Corporation',
        }
        
        cleaned_df['PS_Station'] = cleaned_df['PS_Station'].replace(station_mapping)
    
    # Extract field of interest if available, otherwise derive from PS station
    if 'Field_of_Interest' not in cleaned_df.columns:
        # This is a placeholder. In a real scenario, you would derive this from station characteristics
        # or use NLP to extract from station descriptions if available
        cleaned_df['Field_of_Interest'] = 'Unknown'
    
    # Drop rows with missing critical values
    cleaned_df.dropna(subset=['Student_ID', 'PS_Station'], inplace=True)
    
    # Remove duplicates
    cleaned_df.drop_duplicates(subset=['Student_ID'], keep='first', inplace=True)
    
    # Reset index
    cleaned_df.reset_index(drop=True, inplace=True)
    
    return cleaned_df
