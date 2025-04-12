import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_allocation_by_branch(data):
    """
    Create a visualization showing PS station allocations by branch.
    
    Args:
        data: pandas DataFrame with historical allocation data
        
    Returns:
        plotly.graph_objects.Figure: Interactive visualization
    """
    if 'Branch' not in data.columns or 'PS_Station' not in data.columns:
        # Create a default figure with an error message
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for branch analysis visualization",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get top stations for visualization
    top_stations = data['PS_Station'].value_counts().head(10).index.tolist()
    
    # Filter data for top stations
    filtered_data = data[data['PS_Station'].isin(top_stations)]
    
    # Create a pivot table
    pivot_data = pd.crosstab(filtered_data['Branch'], filtered_data['PS_Station'])
    
    # Create a heatmap visualization
    fig = px.imshow(
        pivot_data,
        labels=dict(x="PS Station", y="Branch", color="Count"),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale="Viridis",
        title="PS Station Allocation by Branch"
    )
    
    # Update layout for better visualization
    fig.update_layout(
        height=600,
        xaxis_tickangle=-45
    )
    
    return fig


def plot_allocation_by_cgpa(data):
    """
    Create a visualization showing CGPA distribution across PS stations.
    
    Args:
        data: pandas DataFrame with historical allocation data
        
    Returns:
        plotly.graph_objects.Figure: Interactive visualization
    """
    if 'CGPA' not in data.columns or 'PS_Station' not in data.columns:
        # Create a default figure with an error message
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for CGPA analysis visualization",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get top stations for visualization
    top_stations = data['PS_Station'].value_counts().head(10).index.tolist()
    
    # Filter data for top stations
    filtered_data = data[data['PS_Station'].isin(top_stations)]
    
    # Create a box plot
    fig = px.box(
        filtered_data,
        x='PS_Station',
        y='CGPA',
        color='PS_Station',
        title="CGPA Distribution Across Top 10 PS Stations",
        labels={'PS_Station': 'PS Station', 'CGPA': 'CGPA'},
    )
    
    # Update layout for better visualization
    fig.update_layout(
        height=600,
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    return fig


def plot_station_popularity(data):
    """
    Create a bar chart showing PS station popularity.
    
    Args:
        data: pandas DataFrame with historical allocation data
        
    Returns:
        plotly.graph_objects.Figure: Interactive visualization
    """
    if 'PS_Station' not in data.columns:
        # Create a default figure with an error message
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for station popularity visualization",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Count occurrences of each station
    station_counts = data['PS_Station'].value_counts().reset_index()
    station_counts.columns = ['PS_Station', 'Count']
    
    # Filter for top 15 stations for visualization
    station_counts = station_counts.head(15)
    
    # Create a bar chart
    fig = px.bar(
        station_counts,
        x='PS_Station',
        y='Count',
        text='Count',
        color='Count',
        color_continuous_scale='Viridis',
        title="Top 15 Most Popular PS Stations",
        labels={'PS_Station': 'PS Station', 'Count': 'Number of Allocations'}
    )
    
    # Update layout for better visualization
    fig.update_layout(
        height=600,
        xaxis_tickangle=-45
    )
    
    return fig


def create_geographical_distribution(data):
    """
    Create a visualization showing the geographical distribution of PS stations.
    
    Args:
        data: pandas DataFrame with historical allocation data
        
    Returns:
        plotly.graph_objects.Figure: Interactive map visualization or None if location data is missing
    """
    # Check if we have location data
    if 'PS_Station' not in data.columns:
        return None
    
    # In a real scenario, you would have latitude and longitude data for each station
    # For this example, we'll use dummy data
    
    # Get unique stations
    stations = data['PS_Station'].unique()
    
    # Create a DataFrame with dummy location data
    # In a real application, you would have a mapping of stations to actual coordinates
    np.random.seed(42)  # For reproducibility
    
    # India's approximate bounding box
    lat_min, lat_max = 8.0, 37.0
    lon_min, lon_max = 68.0, 97.0
    
    location_data = pd.DataFrame({
        'PS_Station': stations,
        'latitude': np.random.uniform(lat_min, lat_max, size=len(stations)),
        'longitude': np.random.uniform(lon_min, lon_max, size=len(stations)),
        'count': [data[data['PS_Station'] == station].shape[0] for station in stations]
    })
    
    # Create a scatter map
    fig = px.scatter_geo(
        location_data,
        lat='latitude',
        lon='longitude',
        hover_name='PS_Station',
        size='count',
        color='count',
        color_continuous_scale='Viridis',
        title="Geographical Distribution of PS Stations",
        projection="natural earth",
        labels={'count': 'Number of Allocations'}
    )
    
    # Focus the map on India
    fig.update_geos(
        visible=False,
        showcountries=True,
        countrycolor="Gray",
        showcoastlines=True,
        coastlinecolor="Gray",
        fitbounds="locations"
    )
    
    return fig
