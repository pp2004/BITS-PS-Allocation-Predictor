import pandas as pd
import numpy as np
import re
from utils import extract_branch_from_id, BRANCH_CODES

def process_excel_data(uploaded_file):
    """
    Process uploaded Excel files containing PS allocation data.
    
    Args:
        uploaded_file: The uploaded Excel file object or path
        
    Returns:
        pandas.DataFrame: Processed data
    """
    try:
        # Check if uploaded_file is a string (file path) or file object
        if isinstance(uploaded_file, str):
            # It's a file path
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            # It's a file object
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Check if the dataframe is empty
        if df.empty:
            print("DataFrame is empty after reading Excel file")
            return None
            
        # Print the columns for debugging
        print(f"Original columns: {df.columns.tolist()}")
            
        # Identify and standardize column names
        df_columns = [str(col).lower() for col in df.columns]
        
        # Map standard column names
        column_mapping = {}
        
        # Student ID column
        for col in df_columns:
            if 'id' in col.lower() or 'number' in col.lower() or 'student' in col.lower() or 'roll' in col.lower():
                column_mapping[df.columns[df_columns.index(col)]] = 'Student_ID'
        
        # CGPA column
        for col in df_columns:
            if 'cgpa' in col.lower() or 'gpa' in col.lower() or 'grade' in col.lower():
                column_mapping[df.columns[df_columns.index(col)]] = 'CGPA'
        
        # PS Station column
        for col in df_columns:
            if 'ps' in col.lower() or 'station' in col.lower() or 'allocation' in col.lower() or 'company' in col.lower():
                column_mapping[df.columns[df_columns.index(col)]] = 'PS_Station'
        
        print(f"Column mapping: {column_mapping}")
        
        # Apply column mapping if found
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # If we still don't have the required columns, try to infer them from the data
        # For example, if there's a column that contains student IDs (like "2022A3PS0558H")
        if 'Student_ID' not in df.columns:
            for col in df.columns:
                # Sample a few values from the column
                sample_values = df[col].astype(str).dropna().head(5).tolist()
                # Check if they look like student IDs
                if any(re.search(r'\d{4}[A-Z]\d[A-Z]{2}\d{4}[A-Z]', str(val)) for val in sample_values):
                    df = df.rename(columns={col: 'Student_ID'})
                    print(f"Inferred Student_ID from column: {col}")
                    break
        
        # Check if required columns exist
        required_columns = ['Student_ID', 'CGPA', 'PS_Station']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        print(f"Missing columns after initial mapping: {missing_columns}")
        
        if missing_columns:
            # Try a more aggressive approach to infer missing columns
            for col in df.columns:
                col_str = str(col).lower()
                
                # Student ID column (broader search)
                if 'Student_ID' in missing_columns and ('id' in col_str or 'number' in col_str or 'student' in col_str or 'roll' in col_str):
                    df = df.rename(columns={col: 'Student_ID'})
                    missing_columns.remove('Student_ID')
                    print(f"Mapped {col} to Student_ID")
                
                # CGPA column (broader search)
                elif 'CGPA' in missing_columns and ('cgpa' in col_str or 'gpa' in col_str or 'grade' in col_str or 'score' in col_str):
                    df = df.rename(columns={col: 'CGPA'})
                    missing_columns.remove('CGPA')
                    print(f"Mapped {col} to CGPA")
                
                # PS Station column (broader search)
                elif 'PS_Station' in missing_columns and ('ps' in col_str or 'station' in col_str or 'company' in col_str or 'org' in col_str or 'alloc' in col_str):
                    df = df.rename(columns={col: 'PS_Station'})
                    missing_columns.remove('PS_Station')
                    print(f"Mapped {col} to PS_Station")
                
                if not missing_columns:
                    break
                
        print(f"Still missing columns: {missing_columns}")
                
        if missing_columns:
            # If still missing columns, try to create them if possible
            if 'Student_ID' in missing_columns:
                # If no student ID column, create a sequential one
                df['Student_ID'] = [f"TEMP{i+1}" for i in range(len(df))]
                missing_columns.remove('Student_ID')
                print("Created Student_ID column with sequential values")
            
            # For CGPA, we can use a default value if not present
            if 'CGPA' in missing_columns:
                df['CGPA'] = 7.5  # Default value
                missing_columns.remove('CGPA')
                print("Created CGPA column with default value of 7.5")
            
            # For PS_Station, if it's missing and we have a column that might contain station names
            if 'PS_Station' in missing_columns:
                # Look for columns that might contain company/organization names
                for col in df.columns:
                    if col != 'Student_ID' and col != 'CGPA':
                        # Check if values look like names of organizations
                        sample_values = df[col].astype(str).dropna().head(5).tolist()
                        # Simple heuristic: if values are mostly strings and not numbers, they might be station names
                        if all(not str(val).replace('.', '').isdigit() for val in sample_values):
                            df = df.rename(columns={col: 'PS_Station'})
                            missing_columns.remove('PS_Station')
                            print(f"Inferred PS_Station from column: {col}")
                            break
            
            # If still missing PS_Station, we cannot proceed
            if 'PS_Station' in missing_columns:
                print("Unable to identify or create PS_Station column")
                return None
        
        # Print final columns
        print(f"Final columns: {df.columns.tolist()}")
        print(f"Sample data:\n{df.head(2)}")
        
        return df
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        import traceback
        traceback.print_exc()
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
