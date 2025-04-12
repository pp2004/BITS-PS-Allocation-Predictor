import re
import pandas as pd
import numpy as np

# Branch code mapping
BRANCH_CODES = {
    'A1 - Civil Engineering': 'A1',
    'A2 - Chemical Engineering': 'A2',
    'A3 - Electrical & Electronics Engineering': 'A3',
    'A4 - Mechanical Engineering': 'A4',
    'A5 - Pharmacy': 'A5',
    'A7 - Computer Science': 'A7',
    'A8 - Electronics & Instrumentation': 'A8',
    'B1 - Manufacturing': 'B1',
    'B2 - Electronics & Communication': 'B2',
    'B3 - Information Systems': 'B3',
    'B4 - Economics & Finance': 'B4',
    'B5 - Biological Sciences': 'B5'
}

# Field options
FIELD_OPTIONS = [
    'Software Development',
    'Data Science',
    'Finance',
    'Consulting',
    'Manufacturing',
    'Healthcare',
    'Research',
    'Infrastructure',
    'Energy',
    'Electronics',
    'Other'
]

def extract_branch_from_id(student_id):
    """
    Extract branch code from student ID.
    
    Args:
        student_id: Student ID string
        
    Returns:
        str: Branch code or 'Unknown' if not identifiable
    """
    if not isinstance(student_id, str):
        return 'Unknown'
    
    # Convert to string and normalize
    student_id = str(student_id).strip().upper()
    
    # Look for branch code patterns in the student ID
    # Try common patterns like: 2022A7PS0123H or 2022B1A70123P
    patterns = [
        r'(\d{4})([AB][1-9])([A-Z]{2})(\d{4})[A-Z]',   # Standard format: 2022A7PS0123H
        r'(\d{4})([AB][1-9])([A-Z])(\d{5})[A-Z]',      # Alternate format: 2022A7P01234H
        r'(\d{4})([AB][1-9])',                         # Simple format: 2022A7
        r'([AB][1-9])'                                 # Just branch code: A7
    ]
    
    for pattern in patterns:
        match = re.search(pattern, student_id)
        if match:
            # Extract the branch code from the matched group
            # The branch code is in the second group for the first two patterns
            # and the first group for the last pattern
            if len(match.groups()) >= 2:
                branch_code = match.group(2)
            else:
                branch_code = match.group(1)
            
            # Map branch code to full branch name
            for branch_name, code in BRANCH_CODES.items():
                if code == branch_code:
                    return branch_name
            
            # If code is found but not in our mapping, update the BRANCH_CODES dictionary
            branch_name = f'{branch_code} - Unknown Program'
            BRANCH_CODES[branch_name] = branch_code
            return branch_name
    
    # If no pattern matched, try to find any branch code-like pattern
    any_branch_match = re.search(r'[AB][1-9]', student_id)
    if any_branch_match:
        branch_code = any_branch_match.group(0)
        branch_name = f'{branch_code} - Unknown Program'
        BRANCH_CODES[branch_name] = branch_code
        return branch_name
    
    return 'Unknown'


def get_detailed_station_info(station_name, historical_data, user_branch):
    """
    Get detailed information about a PS station.
    
    Args:
        station_name: Name of the PS station
        historical_data: DataFrame with historical allocation data
        user_branch: User's branch for personalized insights
        
    Returns:
        dict: Dictionary containing detailed station information
    """
    # Filter historical data for this station
    station_data = historical_data[historical_data['PS_Station'] == station_name]
    
    if station_data.empty:
        return {
            'location': 'Unknown',
            'field': 'Unknown',
            'cgpa_range': 'Unknown',
            'selection_rate': 'Unknown',
            'branch_distribution': None
        }
    
    # Extract location (in a real application, you would have this data)
    location = 'Unknown'
    
    # Determine field (in a real application, you would have this data)
    field = 'Unknown'
    if 'Field_of_Interest' in station_data.columns:
        field = station_data['Field_of_Interest'].mode()[0]
    
    # Calculate CGPA range
    cgpa_range = 'Unknown'
    if 'CGPA' in station_data.columns:
        min_cgpa = station_data['CGPA'].min()
        max_cgpa = station_data['CGPA'].max()
        cgpa_range = f"{min_cgpa:.2f} - {max_cgpa:.2f}"
    
    # Calculate selection rate
    selection_rate = f"{len(station_data)} students"
    
    # Create branch distribution
    branch_distribution = None
    if 'Branch' in station_data.columns:
        branch_counts = station_data['Branch'].value_counts()
        if not branch_counts.empty:
            branch_distribution = branch_counts
    
    return {
        'location': location,
        'field': field,
        'cgpa_range': cgpa_range,
        'selection_rate': selection_rate,
        'branch_distribution': branch_distribution
    }
