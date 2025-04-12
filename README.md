# PS Station Predictor

A machine learning-based Practice School (PS) allocation prediction system that analyzes student profiles and historical data to suggest potential PS stations.

## Overview

This application helps predict potential Practice School (PS) stations based on your academic profile and historical allocation data. The system uses machine learning to analyze patterns in past allocations and suggests stations that might be a good fit for you.

![PS Station Predictor Screenshot](screenshots/app_screenshot.png)

## Features

- **Predictive Analytics**: Uses machine learning to predict PS station allocations based on student profiles
- **Data Visualization**: Interactive charts and graphs to explore allocation trends
- **Surprise Recommendations**: Get unexpected but potentially suitable station recommendations
- **Historical Data Analysis**: Analyze past allocation patterns by branch, CGPA, and more
- **User-friendly Interface**: Easy-to-use Streamlit web interface

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (listed in requirements.txt)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ps-station-predictor.git
   cd ps-station-predictor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python run_local.py
   ```
   
   Or directly with Streamlit:
   ```
   streamlit run app.py
   ```

4. Access the application in your browser at http://localhost:8501

## Usage

1. **Data Upload**: Upload historical PS allocation data from Excel files
2. **Model Training**: Train the prediction model using the uploaded data
3. **Prediction**: Enter your details to get personalized PS station recommendations
4. **Visualizations**: Explore trends and patterns in PS allocations

## Data Format

The application expects Excel files with the following columns:
- Student ID
- CGPA
- PS Station
- Other relevant fields (Branch, Field of Interest, etc.)

Sample data files are included in the `attached_assets` folder.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Developed for IEEE TechWeek Hackathon
- Thanks to all contributors and testers