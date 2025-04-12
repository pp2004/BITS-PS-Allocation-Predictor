import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from data_processor import process_excel_data, clean_data
from model_trainer import train_model
from predictor import predict_ps_stations, get_surprise_recommendation
from visualizer import (
    plot_allocation_by_branch, 
    plot_allocation_by_cgpa, 
    plot_station_popularity,
    create_geographical_distribution
)
from utils import (
    BRANCH_CODES,
    FIELD_OPTIONS,
    extract_branch_from_id,
    get_detailed_station_info
)

# Set page config
st.set_page_config(
    page_title="PS Station Predictor",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'ps_stations' not in st.session_state:
    st.session_state.ps_stations = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None


def main():
    st.title("Practice School Allocation Prediction System")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Data Upload", "Model Training", "Prediction", "Visualizations"]
    )
    
    # Display the selected page
    if page == "Home":
        show_home_page()
    elif page == "Data Upload":
        show_data_upload_page()
    elif page == "Model Training":
        show_model_training_page()
    elif page == "Prediction":
        show_prediction_page()
    elif page == "Visualizations":
        show_visualization_page()


def show_home_page():
    st.header("Welcome to the PS Station Predictor")
    
    st.markdown("""
    This application helps predict potential Practice School (PS) stations based on your academic profile 
    and historical allocation data. The system uses machine learning to analyze patterns in past allocations
    and suggests stations that might be a good fit for you.
    
    ### How to use this application:
    
    1. **Data Upload**: Upload historical PS allocation data from Excel files
    2. **Model Training**: Train the prediction model using the uploaded data
    3. **Prediction**: Enter your details to get personalized PS station recommendations
    4. **Visualizations**: Explore trends and patterns in PS allocations
    
    ### Features:
    
    - Predictive analytics for PS station allocation
    - "Surprise Me" recommendations for unexpected but potentially suitable stations
    - Detailed PS station information and insights
    - Visual analysis of allocation trends by branch, CGPA, and field
    
    Get started by navigating to the Data Upload section using the sidebar.
    """)


def show_data_upload_page():
    st.header("Upload PS Allocation Data")
    
    st.markdown("""
    Upload Excel files containing historical PS allocation data. 
    The system will process and analyze this data to train the prediction model.
    
    **Required Excel format:**
    - Student ID
    - CGPA
    - PS Station
    - Other relevant fields (Branch, Field of Interest, etc.)
    
    Upload both past and current semester data if available.
    """)
    
    # Option to use predefined data files
    use_sample_data = st.checkbox("Use sample data files (faster)")
    
    merged_data = None
    
    if use_sample_data:
        st.write("Using sample data files from attached_assets folder.")
        
        with st.spinner('Processing sample data files...'):
            try:
                # Process the first sample file
                file_path1 = "attached_assets/22-23 SEM2 (1).xlsx"
                st.write(f"Processing file: {file_path1}")
                raw_data1 = process_excel_data(file_path1)
                
                # Process the second sample file
                file_path2 = "attached_assets/24-25 SEM2 PS2.xlsx"
                st.write(f"Processing file: {file_path2}")
                raw_data2 = process_excel_data(file_path2)
                
                # Combine data from both files if they were processed successfully
                if raw_data1 is not None and not raw_data1.empty and raw_data2 is not None and not raw_data2.empty:
                    merged_data = pd.concat([raw_data1, raw_data2], ignore_index=True)
                    st.success("Sample data files processed and merged successfully!")
                elif raw_data1 is not None and not raw_data1.empty:
                    merged_data = raw_data1
                    st.success("First sample data file processed successfully!")
                    st.warning("Second sample data file could not be processed.")
                elif raw_data2 is not None and not raw_data2.empty:
                    merged_data = raw_data2
                    st.success("Second sample data file processed successfully!")
                    st.warning("First sample data file could not be processed.")
                else:
                    st.error("Failed to process both sample data files. Please try uploading your own files.")
            except Exception as e:
                st.error(f"An error occurred while processing sample data files: {str(e)}")
                st.error("Please try uploading your own files.")
    else:
        # User uploads their own file
        uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
        
        if uploaded_file is not None:
            try:
                with st.spinner('Processing data...'):
                    # Process the uploaded Excel file
                    raw_data = process_excel_data(uploaded_file)
                    
                    if raw_data is not None and not raw_data.empty:
                        merged_data = raw_data
                    else:
                        st.error("Failed to process the file. Please ensure it has the required format.")
            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")
    
    # Process the merged data if available
    if merged_data is not None and not merged_data.empty:
        try:
            with st.spinner('Cleaning and preprocessing data...'):
                # Clean and preprocess the data
                cleaned_data = clean_data(merged_data)
                
                # Store the processed data in session state
                st.session_state.historical_data = cleaned_data
                
                # Display preview of the processed data
                st.success("Data cleaned and preprocessed successfully!")
                st.subheader("Preview of Processed Data")
                st.dataframe(cleaned_data.head(10))
                
                # Display some basic statistics
                st.subheader("Data Summary")
                st.write(f"Total Records: {len(cleaned_data)}")
                
                branches = cleaned_data['Branch'].value_counts()
                st.write("Branch Distribution:")
                st.bar_chart(branches)
                
                # Display PS station counts
                station_counts = cleaned_data['PS_Station'].value_counts().head(10)
                st.write("Top 10 Popular PS Stations:")
                st.bar_chart(station_counts)
        except Exception as e:
            st.error(f"An error occurred during data cleaning: {str(e)}")


def show_model_training_page():
    st.header("Train Prediction Model")
    
    if st.session_state.historical_data is None:
        st.warning("Please upload historical data first in the Data Upload section.")
        return
    
    st.markdown("""
    Train the machine learning model using the uploaded historical data.
    This model will be used to predict PS station allocations based on student profiles.
    
    You can select which features to include in the model training.
    """)
    
    # Allow user to select features for training
    available_features = list(st.session_state.historical_data.columns)
    available_features.remove('PS_Station')  # Remove the target variable
    
    selected_features = st.multiselect(
        "Select features to use for prediction",
        available_features,
        default=['Branch', 'CGPA'] if 'Branch' in available_features and 'CGPA' in available_features else []
    )
    
    if st.button("Train Model") and selected_features:
        with st.spinner('Training model... This may take a moment.'):
            # Train the model using selected features
            model, feature_cols, stations = train_model(
                st.session_state.historical_data, 
                selected_features
            )
            
            # Store the trained model and feature columns in session state
            st.session_state.trained_model = model
            st.session_state.feature_columns = feature_cols
            st.session_state.ps_stations = stations
            
            st.success("Model trained successfully!")
            
            # Display feature importance or model details
            st.subheader("Model Information")
            st.write(f"Features used: {', '.join(feature_cols)}")
            st.write(f"Number of PS stations in the model: {len(stations)}")
            
            # If we have feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': [feature_cols[i] for i in indices],
                    'Importance': importances[indices]
                })
                
                fig = px.bar(importance_df, x='Feature', y='Importance')
                st.plotly_chart(fig)
    elif not selected_features and st.button("Train Model"):
        st.warning("Please select at least one feature for model training.")


def show_prediction_page():
    st.header("PS Station Prediction")
    
    if st.session_state.trained_model is None:
        st.warning("Please train a model first in the Model Training section.")
        return
    
    st.markdown("""
    Enter your details to get personalized PS station recommendations.
    The system will predict stations based on your profile and the trained model.
    """)
    
    # User input form
    with st.form("prediction_form"):
        # Branch selection
        branch = st.selectbox("Select your branch", options=list(BRANCH_CODES.keys()))
        
        # CGPA input
        cgpa = st.slider("Enter your CGPA", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
        
        # Field of interest
        field_of_interest = st.selectbox("Select your field of interest", options=FIELD_OPTIONS)
        
        # Form submission button
        submitted = st.form_submit_button("Predict PS Stations")
    
    # Surprise me button
    surprise_me = st.button("Surprise Me! 🎲")
    
    if submitted:
        with st.spinner('Generating predictions...'):
            # Create a user profile based on inputs
            user_profile = {
                'Branch': branch,
                'CGPA': cgpa,
                'Field_of_Interest': field_of_interest
            }
            
            # Get predictions
            predictions = predict_ps_stations(
                user_profile,
                st.session_state.trained_model,
                st.session_state.feature_columns,
                st.session_state.ps_stations,
                st.session_state.historical_data
            )
            
            # Store predictions in session state
            st.session_state.predictions = predictions
            
            # Display predictions
            display_predictions(predictions)
            
    elif surprise_me:
        with st.spinner('Finding a surprise recommendation...'):
            # Create a user profile based on inputs (using default values if not provided)
            user_profile = {
                'Branch': branch if 'branch' in locals() else list(BRANCH_CODES.keys())[0],
                'CGPA': cgpa if 'cgpa' in locals() else 8.0,
                'Field_of_Interest': field_of_interest if 'field_of_interest' in locals() else FIELD_OPTIONS[0]
            }
            
            # Get a surprise recommendation
            surprise_rec = get_surprise_recommendation(
                user_profile,
                st.session_state.trained_model,
                st.session_state.feature_columns,
                st.session_state.ps_stations,
                st.session_state.historical_data
            )
            
            # Display the surprise recommendation
            st.subheader("Surprise Recommendation! 🎉")
            
            if surprise_rec:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### {surprise_rec['station']}")
                    st.write(f"**Location:** {surprise_rec.get('location', 'N/A')}")
                    st.write(f"**Field:** {surprise_rec.get('field', 'N/A')}")
                    st.write(f"**CGPA Range:** {surprise_rec.get('cgpa_range', 'N/A')}")
                    st.write(f"**Selection Rate:** {surprise_rec.get('selection_rate', 'N/A')}")
                
                with col2:
                    st.subheader("Why this might interest you:")
                    st.write(surprise_rec.get('reason', 'This station offers a unique opportunity that may align with your interests in surprising ways.'))
                    
                    st.subheader("Historical Trends:")
                    if 'historical_trend' in surprise_rec and surprise_rec['historical_trend'] is not None:
                        st.line_chart(surprise_rec['historical_trend'])
                    else:
                        st.info("No historical trend data available for this station.")
            else:
                st.info("Couldn't generate a surprise recommendation. Please try again or check your profile details.")


def display_predictions(predictions):
    if not predictions or len(predictions) == 0:
        st.warning("No predictions could be generated. Please try with different profile details.")
        return
    
    st.subheader("Recommended PS Stations")
    
    # Create tabs for Top Picks and All Recommendations
    tab1, tab2 = st.tabs(["Top Picks", "All Recommendations"])
    
    with tab1:
        # Display top 3 recommendations with detailed cards
        for i, pred in enumerate(predictions[:3]):
            with st.container():
                st.markdown(f"### {i+1}. {pred['station']}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Location:** {pred.get('location', 'N/A')}")
                    st.write(f"**Field:** {pred.get('field', 'N/A')}")
                    st.write(f"**CGPA Range:** {pred.get('cgpa_range', 'N/A')}")
                    st.write(f"**Compatibility Score:** {pred.get('score', 'N/A')}")
                    st.write(f"**Selection Rate:** {pred.get('selection_rate', 'N/A')}")
                
                with col2:
                    if 'branch_distribution' in pred and pred['branch_distribution'] is not None:
                        st.write("**Branch Distribution:**")
                        # Convert to dataframe for plotting with plotly
                        if isinstance(pred['branch_distribution'], pd.Series):
                            branch_dist_df = pred['branch_distribution'].reset_index()
                            branch_dist_df.columns = ['Branch', 'Count']
                            fig = px.pie(branch_dist_df, values='Count', names='Branch', title='Branch Distribution')
                            st.plotly_chart(fig)
                
                st.write("---")
    
    with tab2:
        # Display all recommendations in a table
        table_data = []
        for pred in predictions:
            table_data.append({
                "PS Station": pred['station'],
                "Location": pred.get('location', 'N/A'),
                "Field": pred.get('field', 'N/A'),
                "CGPA Range": pred.get('cgpa_range', 'N/A'),
                "Compatibility Score": pred.get('score', 'N/A')
            })
        
        if table_data:
            st.dataframe(pd.DataFrame(table_data))
        else:
            st.info("No additional recommendations available.")


def show_visualization_page():
    st.header("PS Allocation Visualizations")
    
    if st.session_state.historical_data is None:
        st.warning("Please upload historical data first in the Data Upload section.")
        return
    
    st.markdown("""
    Explore trends and patterns in PS allocations through interactive visualizations.
    These insights can help understand allocation patterns and make informed decisions.
    """)
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "Branch Analysis", 
        "CGPA Distribution", 
        "Station Popularity",
        "Geographical Distribution"
    ])
    
    with tab1:
        st.subheader("PS Allocation by Branch")
        fig = plot_allocation_by_branch(st.session_state.historical_data)
        st.plotly_chart(fig)
    
    with tab2:
        st.subheader("CGPA Distribution Across PS Stations")
        fig = plot_allocation_by_cgpa(st.session_state.historical_data)
        st.plotly_chart(fig)
    
    with tab3:
        st.subheader("PS Station Popularity")
        
        # Option to filter by branch
        all_branches = ["All"] + list(st.session_state.historical_data['Branch'].unique())
        selected_branch = st.selectbox("Select Branch", options=all_branches)
        
        # Filter data based on selection
        if selected_branch == "All":
            filtered_data = st.session_state.historical_data
        else:
            filtered_data = st.session_state.historical_data[st.session_state.historical_data['Branch'] == selected_branch]
        
        # Create and display the plot
        fig = plot_station_popularity(filtered_data)
        st.plotly_chart(fig)
    
    with tab4:
        st.subheader("Geographical Distribution of PS Stations")
        
        # Create the geographical visualization
        fig = create_geographical_distribution(st.session_state.historical_data)
        
        if fig is not None:
            st.plotly_chart(fig)
        else:
            st.info("Geographical data not available for visualization.")


if __name__ == "__main__":
    main()
