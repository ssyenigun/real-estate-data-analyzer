import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.repo_manager import RepoManager

st.set_page_config(page_title="Price Distribution Analysis", layout="wide")
st.title("📊 Price Distribution Analysis")
st.markdown("---")

def load_data():
    """Load and preprocess data"""
    try:
        df = pd.read_csv('src/processed_data.csv')
        if 'Property Price' in df.columns:
            df['Property Price'] = pd.to_numeric(df['Property Price'].astype(str).str.replace(',', ''), errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_price_distribution(df):
    """Create price distribution visualization with statistical tests"""
    # Create histogram data
    hist_values, bin_edges = np.histogram(df['Property Price'], bins=50)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
    
    # Create Q-Q plot data
    qq = stats.probplot(df['Property Price'], dist="norm")
    qq_x = qq[0][0].tolist()
    qq_y = qq[0][1].tolist()
    qq_line_x = qq_x
    qq_line_y = (qq[1][0] * np.array(qq_x) + qq[1][1]).tolist()
    
    # Create box plot data
    box_data = df['Property Price'].describe()
    box_stats = [
        box_data['min'],
        box_data['25%'],
        box_data['50%'],
        box_data['75%'],
        box_data['max']
    ]
    
    # Create density plot data
    kde = stats.gaussian_kde(df['Property Price'])
    x_range = np.linspace(df['Property Price'].min(), df['Property Price'].max(), 100)
    density_y = kde(x_range).tolist()
    
    # Create ECharts options
    options = {
        "title": {
            "text": "Price Distribution Analysis",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"}
        },
        "grid": [
            {"left": "5%", "right": "5%", "height": "30%", "top": "5%"},
            {"left": "5%", "right": "5%", "height": "30%", "top": "40%"},
            {"left": "5%", "right": "5%", "height": "30%", "top": "75%"}
        ],
        "xAxis": [
            {
                "type": "category",
                "data": [f"${int(x):,}" for x in bin_centers],
                "name": "Price Range",
                "nameLocation": "middle",
                "nameGap": 30,
                "gridIndex": 0
            },
            {
                "type": "value",
                "name": "Theoretical Quantiles",
                "nameLocation": "middle",
                "nameGap": 30,
                "gridIndex": 1
            },
            {
                "type": "value",
                "name": "Price",
                "nameLocation": "middle",
                "nameGap": 30,
                "gridIndex": 2
            }
        ],
        "yAxis": [
            {
                "type": "value",
                "name": "Count",
                "nameLocation": "middle",
                "nameGap": 30,
                "gridIndex": 0
            },
            {
                "type": "value",
                "name": "Sample Quantiles",
                "nameLocation": "middle",
                "nameGap": 30,
                "gridIndex": 1
            },
            {
                "type": "value",
                "name": "Density",
                "nameLocation": "middle",
                "nameGap": 30,
                "gridIndex": 2
            }
        ],
        "series": [
            {
                "name": "Histogram",
                "type": "bar",
                "data": hist_values.tolist(),
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "itemStyle": {"color": "#2E86C1"}
            },
            {
                "name": "Q-Q Plot",
                "type": "scatter",
                "data": [[x, y] for x, y in zip(qq_x, qq_y)],
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "itemStyle": {"color": "#E74C3C"}
            },
            {
                "name": "Normal Line",
                "type": "line",
                "data": [[x, y] for x, y in zip(qq_line_x, qq_line_y)],
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "itemStyle": {"color": "#27AE60"}
            },
            {
                "name": "Density",
                "type": "line",
                "data": [[x, y] for x, y in zip(x_range, density_y)],
                "xAxisIndex": 2,
                "yAxisIndex": 2,
                "itemStyle": {"color": "#8E44AD"}
            }
        ]
    }
    
    # Perform statistical tests
    shapiro_test = stats.shapiro(df['Property Price'])
    ks_test = stats.kstest(df['Property Price'], 'norm')
    
    # Create statistical summary
    stats_summary = {
        "Shapiro-Wilk Test": {
            "statistic": shapiro_test[0],
            "p-value": shapiro_test[1],
            "interpretation": "Normal distribution" if shapiro_test[1] > 0.05 else "Non-normal distribution"
        },
        "Kolmogorov-Smirnov Test": {
            "statistic": ks_test[0],
            "p-value": ks_test[1],
            "interpretation": "Normal distribution" if ks_test[1] > 0.05 else "Non-normal distribution"
        },
        "Basic Statistics": {
            "mean": np.mean(df['Property Price']),
            "median": np.median(df['Property Price']),
            "std": np.std(df['Property Price']),
            "skewness": stats.skew(df['Property Price']),
            "kurtosis": stats.kurtosis(df['Property Price'])
        }
    }
    
    return options, stats_summary

def create_price_by_property_type(df):
    """Create price by property type visualization"""
    if 'Property Type' not in df.columns or 'Property Price' not in df.columns:
        return None
        
    type_avg = df.groupby('Property Type')['Property Price'].mean().sort_values(ascending=False)
    
    option = {
        "title": {
            "text": "Average Property Price by Type",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
        },
        "xAxis": {
            "type": "category",
            "data": type_avg.index.tolist(),
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {
            "type": "value",
            "name": "Average Price ($)",
            "nameLocation": "middle",
            "nameGap": 70,
            "axisLabel": {"formatter": "${value}"}
        },
        "series": [{
            "data": type_avg.values.tolist(),
            "type": "bar",
            "itemStyle": {"color": "#F1C40F"}
        }],
        "grid": {"bottom": "15%"}
    }
    return option

def create_price_by_bedrooms(df):
    """Create bedroom count vs property price visualization"""
    if 'Bedrooms' not in df.columns or 'Property Price' not in df.columns:
        return None
        
    bedroom_avg = df.groupby('Bedrooms')['Property Price'].agg(['mean','count']).reset_index()
    bedroom_avg = bedroom_avg[bedroom_avg['count'] >= 5]
    
    option = {
        "title": {
            "text": "Average Property Price by Number of Bedrooms",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
        },
        "xAxis": {
            "type": "category",
            "data": bedroom_avg['Bedrooms'].tolist(),
            "name": "Number of Bedrooms",
            "nameLocation": "middle",
            "nameGap": 30
        },
        "yAxis": {
            "type": "value",
            "name": "Average Price ($)",
            "nameLocation": "middle",
            "nameGap": 70,
            "axisLabel": {"formatter": "${value}"}
        },
        "series": [{
            "data": bedroom_avg['mean'].tolist(),
            "type": "bar",
            "itemStyle": {"color": "#9B59B6"}
        }]
    }
    return option

def create_price_by_city(df):
    """Create price by city visualization"""
    if 'City' not in df.columns or 'Property Price' not in df.columns:
        return None
    
    city_avg = df.groupby('City')['Property Price'].mean().sort_values(ascending=False).head(10)
    
    option = {
        "title": {
            "text": "Top 10 Cities by Average Property Price",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
            "formatter": "City: {b}<br/>Avg Price: ${c:,.2f}"
        },
        "xAxis": {
            "type": "category",
            "data": city_avg.index.tolist(),
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {
            "type": "value",
            "name": "Average Price ($)",
            "axisLabel": {"formatter": "${value}"}
        },
        "series": [{
            "data": city_avg.values.tolist(),
            "type": "bar",
            "itemStyle": {"color": "#3498DB"}
        }],
        "grid": {"bottom": "15%"}
    }
    return option

def main():
    # Load data
    df = load_data()
    if df is None:
        return

    st.header("Price Analysis Outlier Detect")
    
    # First row - Price Distribution and Property Type
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Price Distribution")
        st_echarts(create_price_distribution(df)[0], height="400px")
    
    with col2:
        st.subheader("🏠 Price by Property Type")
        st_echarts(create_price_by_property_type(df), height="400px")
    
    # Second row - Price by Bedrooms
    st.subheader("🛏️ Price by Number of Bedrooms")
    st_echarts(create_price_by_bedrooms(df), height="400px")
    
    # Third row - Price by City
    st.subheader("🌆 Price by City")
    st_echarts(create_price_by_city(df), height="400px")
    
    # Statistical Analysis Section
    st.markdown("---")
    st.subheader("📊 Statistical Analysis")
    
    # Get statistical summary
    _, stats_summary = create_price_distribution(df)
    
    # Display statistical tests
    st.write("### Shapiro-Wilk Test")
    st.write(f"Statistic: {stats_summary['Shapiro-Wilk Test']['statistic']:.4f}")
    st.write(f"p-value: {stats_summary['Shapiro-Wilk Test']['p-value']:.4f}")
    st.write(f"Interpretation: {stats_summary['Shapiro-Wilk Test']['interpretation']}")
    
    st.write("### Kolmogorov-Smirnov Test")
    st.write(f"Statistic: {stats_summary['Kolmogorov-Smirnov Test']['statistic']:.4f}")
    st.write(f"p-value: {stats_summary['Kolmogorov-Smirnov Test']['p-value']:.4f}")
    st.write(f"Interpretation: {stats_summary['Kolmogorov-Smirnov Test']['interpretation']}")
    
    st.write("### Basic Statistics")
    st.write(f"Mean: ${stats_summary['Basic Statistics']['mean']:,.2f}")
    st.write(f"Median: ${stats_summary['Basic Statistics']['median']:,.2f}")
    st.write(f"Standard Deviation: ${stats_summary['Basic Statistics']['std']:,.2f}")
    st.write(f"Skewness: {stats_summary['Basic Statistics']['skewness']:.4f}")
    st.write(f"Kurtosis: {stats_summary['Basic Statistics']['kurtosis']:.4f}")
    
    # Add interpretation section
    st.subheader("📝 Interpretation")
    st.write("""
    The price distribution analysis helps us understand:
    1. The shape of the price distribution
    2. Whether the prices follow a normal distribution
    3. The presence of outliers
    4. The central tendency and spread of prices
    
    The statistical tests help determine if the price distribution is normal:
    - If p-value > 0.05: The distribution is likely normal
    - If p-value ≤ 0.05: The distribution is likely non-normal
    
    Skewness indicates the asymmetry of the distribution:
    - Positive skewness: Right-skewed distribution
    - Negative skewness: Left-skewed distribution
    
    Kurtosis indicates the "tailedness" of the distribution:
    - Positive kurtosis: Heavy tails
    - Negative kurtosis: Light tails
    """)

if __name__ == "__main__":
    main() 