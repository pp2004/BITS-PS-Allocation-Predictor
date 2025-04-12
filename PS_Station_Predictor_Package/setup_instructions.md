# PS Station Predictor - Setup Instructions

This document provides instructions for setting up and running the PS Station Predictor application on your local machine.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Setup Instructions

### 1. Clone or Download the Repository

First, download all the project files to your local machine.

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies

```bash
pip install streamlit pandas numpy matplotlib plotly scikit-learn openpyxl
```

### 4. Create the Streamlit Configuration Directory

Create a `.streamlit` directory in your project folder if it doesn't exist:

```bash
mkdir -p .streamlit
```

### 5. Create or Update the Streamlit Configuration

Create a file named `config.toml` in the `.streamlit` directory with the following content:

```toml
[server]
headless = false
port = 8501
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application should now open in your default web browser at `http://localhost:8501`.

## Project Structure

- `app.py`: Main Streamlit application file
- `data_processor.py`: Functions for processing and cleaning data
- `model_trainer.py`: Machine learning model training functionality
- `predictor.py`: Prediction and recommendation logic
- `visualizer.py`: Data visualization functions
- `utils.py`: Utility functions and constants
- `attached_assets/`: Directory containing sample Excel files

## Usage

1. Start by navigating to the "Data Upload" section in the sidebar
2. Upload an Excel file with PS allocation data or use the sample data files
3. Go to the "Model Training" section to train a prediction model
4. Use the "Prediction" section to get personalized PS station recommendations
5. Try the "Surprise Me" feature for unexpected but potentially suitable recommendations
6. Explore the "Visualizations" section to analyze allocation trends

## Customization

You can customize the application by modifying:

- `utils.py`: Update the branch code mappings or field options
- `visualizer.py`: Customize visualization styles and types
- `predictor.py`: Adjust the prediction and recommendation algorithms

## Troubleshooting

If you encounter issues:

1. Make sure all the required dependencies are installed
2. Check that you have the latest version of Python and pip
3. Ensure the Excel files are in the correct format
4. Try restarting the application after making any code changes

## Sample Data Files

The repository includes two sample Excel files in the `attached_assets` directory:
- `22-23 SEM2 (1).xlsx`
- `24-25 SEM2 PS2.xlsx`

These files can be used to test the application without uploading your own data.