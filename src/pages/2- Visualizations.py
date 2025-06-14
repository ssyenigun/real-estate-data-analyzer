import streamlit as st
import pandas as pd
import numpy as np
import re
import sys
import os
from streamlit_echarts import st_echarts
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json

# src klasörünü Python path'ine ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.repo_manager import RepoManager

st.set_page_config(page_title="Real Estate Data Analysis - Visualizations", layout="wide")
st.title("📊 Data Visualizations")
st.markdown("---")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Repository manager
repo = RepoManager()

# Helper functions for data processing
def extract_numeric_price(price_str):
    """Extract numeric price from string"""
    if pd.isna(price_str):
        return None
    price_str = str(price_str).replace('USD $', '').replace(',', '').replace(' per month', '')
    try:
        return float(price_str)
    except:
        return None

def extract_numeric_size(size_str):
    """Extract numeric size from string"""
    if pd.isna(size_str):
        return None
    size_str = str(size_str).replace(',', '').replace(' sq ft', '')
    try:
        return float(size_str)
    except:
        return None

def extract_bedrooms(room_str):
    """Extract bedroom count from room string"""
    if pd.isna(room_str):
        return 0
    match = re.search(r'(\d+) bedroom', str(room_str))
    return int(match.group(1)) if match else 0

def preprocess_data(df):
    """Preprocess raw data for visualization"""
    df_processed = df.copy()
    
    # Convert price column
    if 'Property Price' in df_processed.columns:
        df_processed['Numeric_Price'] = df_processed['Property Price'].apply(extract_numeric_price)
    
    # Convert size column
    if 'Building Size' in df_processed.columns:
        df_processed['Numeric_Size'] = df_processed['Building Size'].apply(extract_numeric_size)
    
    # Extract bedrooms
    if 'Rooms' in df_processed.columns:
        df_processed['Bedrooms'] = df_processed['Rooms'].apply(extract_bedrooms)
    
    # Convert date column
    date_columns = ['Published_Date', 'Published On', 'Published Date']
    for col in date_columns:
        if col in df_processed.columns:
            try:
                df_processed['Published_Date'] = pd.to_datetime(df_processed[col], format='%d %b %Y')
                break
            except:
                continue
    
    return df_processed

def load_data():
    """Load the preprocessed data"""
    try:
        # First try to get data from session state
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            return st.session_state.processed_data
        if 'raw_data' in st.session_state and st.session_state.raw_data is not None:
            return st.session_state.raw_data
            
        # If no data in session state, try to load from file
        possible_paths = [
            Path('preprocessed_data.csv'),
            Path('src/preprocessed_data.csv'),
            Path('src/data/preprocessed_data.csv'),
            Path('data/preprocessed_data.csv')
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            st.error("""
            ❌ No data found. Please follow these steps:
            
            1. Go to the "Data Preparation" page (first page in the sidebar)
            2. Upload your Excel file (.xlsx or .xls)
            3. Process the data using the provided options
            4. Return to this page to view visualizations
            
            The data file should be in Excel format and contain at least these columns:
            - Property Price
            - Building Size
            - Rooms
            - Property Type
            - City
            """)
            return None
        
        data = pd.read_csv(data_path)
        if data.empty:
            st.error("❌ The data file is empty.")
            return None
            
        st.success(f"✅ Data loaded successfully from {data_path}. Shape: {data.shape}")
        return data
    except Exception as e:
        st.error(f"""
        ❌ Error loading data: {str(e)}
        
        Please make sure:
        1. The data file exists in one of the expected locations
        2. The file is a valid CSV file
        3. You have uploaded data in the Data Upload page first
        """)
        return None

# Load data if not in session state
if 'processed_data' not in st.session_state and 'raw_data' not in st.session_state:
    df = load_data()
    if df is not None:
        st.session_state.raw_data = df
        st.session_state.processed_data = preprocess_data(df)
    else:
        st.error("❌ Please upload your data in the Data Upload page first.")
        st.stop()

# Check if data exists and handle data selection
if 'processed_data' in st.session_state or 'raw_data' in st.session_state:
    # Data selection
    data_type = st.sidebar.selectbox(
        "Select Data Type",
        ["Processed Data", "Raw Data"],
        help="Choose between processed data (cleaned and transformed) or raw data (original format)"
    )
    
    if data_type == "Processed Data" and 'processed_data' in st.session_state:
        df = st.session_state.processed_data
        st.success("✅ Using processed data")
    elif data_type == "Raw Data" and 'raw_data' in st.session_state:
        df = st.session_state.raw_data
        st.warning("⚠️ Using raw data")
    else:
        st.error("❌ Selected data type not available. Please process data first.")
        st.stop()
else:
    st.error("❌ No data found in session. Please upload and process data first.")
    st.stop()

# Preprocess data if using raw data
if data_type == "Raw Data":
    df = preprocess_data(df)

# Sidebar - Visualization Options
st.sidebar.header("📊 Visualization Options")

viz_categories = {
    "Price Analysis": ["Price Distribution", "Price by Property Type", "Price vs Size", "City Price Comparison"],
    "Property Analysis": ["Property Type Distribution", "Room Distribution", "Size Analysis", "Building Age Analysis"],
    "Location Analysis": ["City Distribution", "Crime Rate Analysis", "Geographic Insights"],
    "Advanced Analysis": ["Market Trends", "Price Correlation", "Statistical Analysis"]
}

selected_category = st.sidebar.selectbox("Select Category", list(viz_categories.keys()))
selected_viz = st.sidebar.selectbox("Select Visualization", viz_categories[selected_category])

# Filter options
st.sidebar.header("🔍 Filters")
if 'Property Type' in df.columns:
    property_types = st.sidebar.multiselect(
        "Property Types",
        df['Property Type'].unique(),
        default=df['Property Type'].unique()
    )
    df_filtered = df[df['Property Type'].isin(property_types)]
else:
    df_filtered = df

if 'City' in df.columns:
    cities = st.sidebar.multiselect(
        "Cities",
        df['City'].unique(),
        default=df['City'].unique()[:10] if len(df['City'].unique()) > 10 else df['City'].unique()
    )
    df_filtered = df_filtered[df_filtered['City'].isin(cities)]

# Main visualization area
col1, col2 = st.columns([3, 1])

with col1:
    try:
        if selected_viz == "Price Distribution":
            st.subheader("💰 Price Distribution")
            
            # Use Numeric_Price if available, otherwise extract from Property Price
            if 'Numeric_Price' in df_filtered.columns:
                prices = df_filtered['Numeric_Price'].dropna().tolist()
            else:
                prices = []
                for price in df_filtered['Property Price'].dropna():
                    numeric_price = extract_numeric_price(price)
                    if numeric_price:
                        prices.append(numeric_price)
            
            if prices:
                # Create histogram data
                hist, bin_edges = np.histogram(prices, bins=30)
                bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
                
                # ECharts histogram
                option = {
                    "title": {"text": "Property Price Distribution", "left": "center"},
                    "tooltip": {"trigger": "axis"},
                    "xAxis": {
                        "type": "category",
                        "data": [f"${int(x):,}" for x in bin_centers],
                        "axisLabel": {"rotate": 45}
                    },
                    "yAxis": {"type": "value", "name": "Count"},
                    "series": [{
                        "data": hist.tolist(),
                        "type": "bar",
                        "itemStyle": {"color": "#5470c6"},
                        "emphasis": {"itemStyle": {"color": "#91cc75"}}
                    }],
                    "grid": {"bottom": "15%"}
                }
                st_echarts(options=option, height="400px")
                
                # Statistics
                st.write("**Price Statistics:**")
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Mean", f"${np.mean(prices):,.0f}")
                with col_b:
                    st.metric("Median", f"${np.median(prices):,.0f}")
                with col_c:
                    st.metric("Min", f"${min(prices):,.0f}")
                with col_d:
                    st.metric("Max", f"${max(prices):,.0f}")
        
        elif selected_viz == "Property Type Distribution":
            st.subheader("🏠 Property Type Distribution")
            
            if 'Property Type' in df_filtered.columns:
                type_counts = df_filtered['Property Type'].value_counts()
                
                # Pie chart
                pie_data = [{"value": count, "name": ptype} for ptype, count in type_counts.items()]
                
                option = {
                    "title": {"text": "Property Type Distribution", "left": "center"},
                    "tooltip": {"trigger": "item"},
                    "legend": {"orient": "vertical", "left": "left"},
                    "series": [{
                        "name": "Property Type",
                        "type": "pie",
                        "radius": "50%",
                        "data": pie_data,
                        "emphasis": {
                            "itemStyle": {
                                "shadowBlur": 10,
                                "shadowOffsetX": 0,
                                "shadowColor": "rgba(0, 0, 0, 0.5)"
                            }
                        }
                    }]
                }
                st_echarts(options=option, height="400px")
                
                # Bar chart
                option2 = {
                    "title": {"text": "Property Type Counts", "left": "center"},
                    "tooltip": {"trigger": "axis"},
                    "xAxis": {
                        "type": "category",
                        "data": type_counts.index.tolist(),
                        "axisLabel": {"rotate": 45}
                    },
                    "yAxis": {"type": "value", "name": "Count"},
                    "series": [{
                        "data": type_counts.values.tolist(),
                        "type": "bar",
                        "itemStyle": {"color": "#91cc75"}
                    }],
                    "grid": {"bottom": "15%"}
                }
                st_echarts(options=option2, height="400px")
            else:
                st.warning("Property Type column not found in the data.")
        
        elif selected_viz == "City Distribution":
            st.subheader("🌆 City Distribution")
            
            if 'City' in df_filtered.columns:
                city_counts = df_filtered['City'].value_counts().head(15)
                
                option = {
                    "title": {"text": "Top 15 Cities by Property Count", "left": "center"},
                    "tooltip": {"trigger": "axis"},
                    "xAxis": {"type": "value", "name": "Count"},
                    "yAxis": {
                        "type": "category",
                        "data": city_counts.index.tolist()[::-1],
                        "axisLabel": {"interval": 0}
                    },
                    "series": [{
                        "data": city_counts.values.tolist()[::-1],
                        "type": "bar",
                        "itemStyle": {"color": "#fac858"}
                    }],
                    "grid": {"left": "20%"}
                }
                st_echarts(options=option, height="600px")
        
        elif selected_viz == "Room Distribution":
            st.subheader("🛏️ Room Distribution")
            
            if 'Rooms' in df_filtered.columns:
                bedroom_counts = []
                for room in df_filtered['Rooms'].dropna():
                    bedrooms = extract_bedrooms(room)
                    bedroom_counts.append(bedrooms)
                
                if bedroom_counts:
                    bedroom_df = pd.DataFrame({'Bedrooms': bedroom_counts})
                    bedroom_dist = bedroom_df['Bedrooms'].value_counts().sort_index()
                    
                    option = {
                        "title": {"text": "Distribution of Bedrooms", "left": "center"},
                        "tooltip": {"trigger": "axis"},
                        "xAxis": {
                            "type": "category",
                            "data": [str(x) for x in bedroom_dist.index]
                        },
                        "yAxis": {"type": "value", "name": "Count"},
                        "series": [{
                            "data": bedroom_dist.values.tolist(),
                            "type": "bar",
                            "itemStyle": {"color": "#ee6666"}
                        }]
                    }
                    st_echarts(options=option, height="400px")
        
        elif selected_viz == "Size Analysis":
            st.subheader("📐 Building Size Analysis")
            
            if 'Building Size' in df_filtered.columns:
                sizes = []
                for size in df_filtered['Building Size'].dropna():
                    numeric_size = extract_numeric_size(size)
                    if numeric_size:
                        sizes.append(numeric_size)
                
                if sizes:
                    # Create histogram
                    hist, bin_edges = np.histogram(sizes, bins=25)
                    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
                    
                    option = {
                        "title": {"text": "Building Size Distribution", "left": "center"},
                        "tooltip": {"trigger": "axis"},
                        "xAxis": {
                            "type": "category",
                            "data": [f"{int(x):,}" for x in bin_centers],
                            "name": "Size (sq ft)",
                            "axisLabel": {"rotate": 45}
                        },
                        "yAxis": {"type": "value", "name": "Count"},
                        "series": [{
                            "data": hist.tolist(),
                            "type": "bar",
                            "itemStyle": {"color": "#73c0de"}
                        }],
                        "grid": {"bottom": "15%"}
                    }
                    st_echarts(options=option, height="400px")
                    
                    # Size statistics
                    st.write("**Size Statistics:**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Average", f"{np.mean(sizes):,.0f} sq ft")
                    with col_b:
                        st.metric("Median", f"{np.median(sizes):,.0f} sq ft")
                    with col_c:
                        st.metric("Max", f"{max(sizes):,.0f} sq ft")
        
        elif selected_viz == "Price vs Size":
            st.subheader("💰📐 Price vs Size Analysis")
            
            # Extract both price and size data
            price_size_data = []
            for _, row in df_filtered.iterrows():
                price = extract_numeric_price(row.get('Property Price'))
                size = extract_numeric_size(row.get('Building Size'))
                if price and size:
                    price_size_data.append([size, price])
            
            if price_size_data:
                option = {
                    "title": {"text": "Property Price vs Building Size", "left": "center"},
                    "tooltip": {
                        "trigger": "item",
                        "formatter": "Size: {c[0]} sq ft<br/>Price: ${c[1]:,}"
                    },
                    "xAxis": {
                        "type": "value",
                        "name": "Building Size (sq ft)",
                        "axisLabel": {"formatter": "{value}"}
                    },
                    "yAxis": {
                        "type": "value",
                        "name": "Price (USD)",
                        "axisLabel": {"formatter": "${value}"}
                    },
                    "series": [{
                        "symbolSize": 8,
                        "data": price_size_data,
                        "type": "scatter",
                        "itemStyle": {"color": "#fc8452"}
                    }]
                }
                st_echarts(options=option, height="500px")
        
        elif selected_viz == "Crime Rate Analysis":
            st.subheader("🚨 Crime Rate Analysis")
            
            if 'Total_Crime_Rate' in df_filtered.columns:
                crime_data = df_filtered[['City', 'Total_Crime_Rate']].dropna()
                if not crime_data.empty:
                    city_crime = crime_data.groupby('City')['Total_Crime_Rate'].mean().sort_values(ascending=False).head(15)
                    
                    option = {
                        "title": {"text": "Average Crime Rate by City", "left": "center"},
                        "tooltip": {"trigger": "axis"},
                        "xAxis": {
                            "type": "category",
                            "data": city_crime.index.tolist(),
                            "axisLabel": {"rotate": 45}
                        },
                        "yAxis": {"type": "value", "name": "Crime Rate"},
                        "series": [{
                            "data": city_crime.values.tolist(),
                            "type": "bar",
                            "itemStyle": {"color": "#d14a61"}
                        }],
                        "grid": {"bottom": "20%"}
                    }
                    st_echarts(options=option, height="500px")
        
        elif selected_viz == "City Price Comparison":
            st.subheader("🌆💰 City Price Comparison")
            
            if 'City' in df_filtered.columns and 'Property Price' in df_filtered.columns:
                city_prices = []
                for city in df_filtered['City'].unique():
                    city_data = df_filtered[df_filtered['City'] == city]
                    prices = []
                    for price in city_data['Property Price'].dropna():
                        numeric_price = extract_numeric_price(price)
                        if numeric_price:
                            prices.append(numeric_price)
                    
                    if prices:
                        city_prices.append({
                            'city': city,
                            'avg_price': np.mean(prices),
                            'count': len(prices)
                        })
                
                if city_prices:
                    city_prices = sorted(city_prices, key=lambda x: x['avg_price'], reverse=True)[:15]
                    
                    cities = [item['city'] for item in city_prices]
                    avg_prices = [item['avg_price'] for item in city_prices]
                    
                    option = {
                        "title": {"text": "Average Property Price by City", "left": "center"},
                        "tooltip": {
                            "trigger": "axis",
                            "formatter": "{b}<br/>Avg Price: ${c:,}"
                        },
                        "xAxis": {
                            "type": "category",
                            "data": cities,
                            "axisLabel": {"rotate": 45}
                        },
                        "yAxis": {
                            "type": "value",
                            "name": "Average Price (USD)",
                            "axisLabel": {"formatter": "${value}"}
                        },
                        "series": [{
                            "data": avg_prices,
                            "type": "bar",
                            "itemStyle": {
                                "color": {
                                    "type": "linear",
                                    "x": 0, "y": 0, "x2": 0, "y2": 1,
                                    "colorStops": [
                                        {"offset": 0, "color": "#83bff6"},
                                        {"offset": 0.5, "color": "#188df0"},
                                        {"offset": 1, "color": "#188df0"}
                                    ]
                                }
                            }
                        }],
                        "grid": {"bottom": "20%"}
                    }
                    st_echarts(options=option, height="500px")
        
        elif selected_viz == "Market Trends":
            st.subheader("📈 Market Trends")
            
            if 'Published_Date' in df_filtered.columns or 'Published On' in df_filtered.columns:
                date_col = 'Published_Date' if 'Published_Date' in df_filtered.columns else 'Published On'
                
                # Convert dates and extract monthly data
                df_dates = df_filtered[df_filtered[date_col].notna()].copy()
                if not df_dates.empty:
                    try:
                        if 'Published_Date' not in df_dates.columns:
                            df_dates['Published_Date'] = pd.to_datetime(df_dates[date_col], format='%d %b %Y')
                        
                        df_dates['Month'] = df_dates['Published_Date'].dt.to_period('M')
                        monthly_counts = df_dates['Month'].value_counts().sort_index()
                        
                        months = [str(month) for month in monthly_counts.index]
                        counts = monthly_counts.values.tolist()
                        
                        option = {
                            "title": {"text": "Property Listings Over Time", "left": "center"},
                            "tooltip": {"trigger": "axis"},
                            "xAxis": {
                                "type": "category",
                                "data": months,
                                "axisLabel": {"rotate": 45}
                            },
                            "yAxis": {"type": "value", "name": "Number of Listings"},
                            "series": [{
                                "data": counts,
                                "type": "line",
                                "smooth": True,
                                "itemStyle": {"color": "#5470c6"},
                                "areaStyle": {"opacity": 0.3}
                            }],
                            "grid": {"bottom": "15%"}
                        }
                        st_echarts(options=option, height="400px")
                    except Exception as e:
                        st.error(f"Error processing dates: {str(e)}")
        
        elif selected_viz == "Price Correlation":
            st.subheader("🔗 Price Correlation Analysis")
            
            # Create correlation data
            correlation_data = []
            
            for _, row in df_filtered.iterrows():
                price = extract_numeric_price(row.get('Property Price'))
                size = extract_numeric_size(row.get('Building Size'))
                bedrooms = extract_bedrooms(row.get('Rooms', ''))
                crime_rate = row.get('Total_Crime_Rate')
                
                if price and size:
                    correlation_data.append({
                        'price': price,
                        'size': size,
                        'bedrooms': bedrooms,
                        'crime_rate': crime_rate if pd.notna(crime_rate) else 0
                    })
            
            if correlation_data:
                corr_df = pd.DataFrame(correlation_data)
                corr_matrix = corr_df.corr()
                
                # Prepare heatmap data
                heatmap_data = []
                columns = corr_matrix.columns.tolist()
                
                for i, row_name in enumerate(corr_matrix.index):
                    for j, col_name in enumerate(columns):
                        heatmap_data.append([j, i, round(corr_matrix.iloc[i, j], 2)])
                
                option = {
                    "title": {"text": "Correlation Matrix", "left": "center"},
                    "tooltip": {
                        "position": "top",
                        "formatter": "Correlation: {c[2]}"
                    },
                    "grid": {"height": "50%", "top": "10%"},
                    "xAxis": {
                        "type": "category",
                        "data": columns,
                        "splitArea": {"show": True}
                    },
                    "yAxis": {
                        "type": "category",
                        "data": columns,
                        "splitArea": {"show": True}
                    },
                    "visualMap": {
                        "min": -1,
                        "max": 1,
                        "calculable": True,
                        "orient": "horizontal",
                        "left": "center",
                        "bottom": "15%",
                        "inRange": {"color": ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]}
                    },
                    "series": [{
                        "name": "Correlation",
                        "type": "heatmap",
                        "data": heatmap_data,
                        "label": {"show": True},
                        "emphasis": {
                            "itemStyle": {
                                "shadowBlur": 10,
                                "shadowColor": "rgba(0, 0, 0, 0.5)"
                            }
                        }
                    }]
                }
                st_echarts(options=option, height="500px")
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.error("Please check if the data contains the required columns and try again.")

with col2:
    st.markdown("### 📊 Data Summary")
    
    # Basic statistics
    st.metric("Total Properties", len(df_filtered))
    
    if 'Property Type' in df_filtered.columns:
        st.metric("Property Types", df_filtered['Property Type'].nunique())
    
    if 'City' in df_filtered.columns:
        st.metric("Cities", df_filtered['City'].nunique())
    
    # Data quality indicators
    st.markdown("### 🔍 Data Quality")
    
    total_rows = len(df_filtered)
    missing_data = df_filtered.isnull().sum()
    
    for col in ['Property Price', 'Building Size', 'Rooms']:
        if col in df_filtered.columns:
            missing_pct = (missing_data[col] / total_rows) * 100
            st.progress(1 - missing_pct/100)
            st.caption(f"{col}: {missing_pct:.1f}% missing")
    
    # Export options
    st.markdown("### 💾 Export")
    
    if st.button("📊 Export Current View"):
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"filtered_data_{selected_viz.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    # Chart customization
    st.markdown("### 🎨 Chart Options")
    
    chart_theme = st.selectbox(
        "Chart Theme",
        ["default", "dark", "vintage", "westeros", "essos", "wonderland", "walden"]
    )
    
    show_animation = st.checkbox("Enable Animation", value=True)
    
    # Quick insights
    st.markdown("### 💡 Quick Insights")
    
    if 'Property Price' in df_filtered.columns:
        prices = [extract_numeric_price(p) for p in df_filtered['Property Price'].dropna()]
        prices = [p for p in prices if p]
        
        if prices:
            avg_price = np.mean(prices)
            st.info(f"💰 Average price: ${avg_price:,.0f}")
            
            if len(prices) > 1:
                price_std = np.std(prices)
                cv = (price_std / avg_price) * 100
                st.info(f"📊 Price variability: {cv:.1f}%")

# Footer
st.markdown("---")
st.markdown("*Interactive charts powered by Apache ECharts. Use filters to explore different data segments.*")

# Add custom CSS for better styling
st.markdown("""
<style>
.metric-container {
  background-color: #f0f2f6;
  padding: 1rem;
  border-radius: 0.5rem;
  margin: 0.5rem 0;
}

.chart-container {
  border: 1px solid #e0e0e0;
  border-radius: 0.5rem;
  padding: 1rem;
  margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def create_price_distribution(df):
    """Create price distribution histogram"""
    hist_values, bin_edges = np.histogram(df['Property Price'], bins=50)
    
    options = {
        "title": {
            "text": "Distribution of Property Prices",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"}
        },
        "xAxis": {
            "type": "category",
            "data": [f"${int(bin_edges[i])}-${int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],
            "name": "Price Range",
            "nameLocation": "middle",
            "nameGap": 30
        },
        "yAxis": {
            "type": "value",
            "name": "Number of Properties",
            "nameLocation": "middle",
            "nameGap": 30
        },
        "series": [{
            "data": hist_values.tolist(),
            "type": "bar",
            "itemStyle": {"color": "#2E86C1"}
        }]
    }
    return options

def create_price_by_property_type(df):
    """Create price by property type box plot"""
    property_types = df['Property Type'].unique()
    data = []
    
    for prop_type in property_types:
        prices = df[df['Property Type'] == prop_type]['Property Price'].tolist()
        data.append({
            "name": prop_type,
            "value": prices
        })
    
    options = {
        "title": {
            "text": "Property Prices by Type",
            "left": "center"
        },
        "tooltip": {
            "trigger": "item",
            "axisPointer": {"type": "shadow"}
        },
        "xAxis": {
            "type": "category",
            "data": property_types.tolist(),
            "name": "Property Type",
            "nameLocation": "middle",
            "nameGap": 30
        },
        "yAxis": {
            "type": "value",
            "name": "Price ($)",
            "nameLocation": "middle",
            "nameGap": 30
        },
        "series": [{
            "type": "boxplot",
            "data": data,
            "itemStyle": {"color": "#2E86C1"}
        }]
    }
    return options

def create_price_by_bedrooms(df):
    """Create price by number of bedrooms box plot"""
    bedrooms = sorted(df['Bedrooms'].unique())
    data = []
    
    for bed in bedrooms:
        prices = df[df['Bedrooms'] == bed]['Property Price'].tolist()
        data.append({
            "name": f"{bed} Bedrooms",
            "value": prices
        })
    
    options = {
        "title": {
            "text": "Property Prices by Number of Bedrooms",
            "left": "center"
        },
        "tooltip": {
            "trigger": "item",
            "axisPointer": {"type": "shadow"}
        },
        "xAxis": {
            "type": "category",
            "data": [f"{bed} Bedrooms" for bed in bedrooms],
            "name": "Number of Bedrooms",
            "nameLocation": "middle",
            "nameGap": 30
        },
        "yAxis": {
            "type": "value",
            "name": "Price ($)",
            "nameLocation": "middle",
            "nameGap": 30
        },
        "series": [{
            "type": "boxplot",
            "data": data,
            "itemStyle": {"color": "#2E86C1"}
        }]
    }
    return options

def create_price_by_city(df):
    """Create price by city bar chart"""
    city_avg = df.groupby('City')['Property Price'].mean().sort_values(ascending=False)
    
    options = {
        "title": {
            "text": "Average Property Prices by City",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"}
        },
        "xAxis": {
            "type": "category",
            "data": city_avg.index.tolist(),
            "name": "City",
            "nameLocation": "middle",
            "nameGap": 30
        },
        "yAxis": {
            "type": "value",
            "name": "Average Price ($)",
            "nameLocation": "middle",
            "nameGap": 30
        },
        "series": [{
            "data": city_avg.values.tolist(),
            "type": "bar",
            "itemStyle": {
                "color": {
                    "type": "linear",
                    "x": 0,
                    "y": 0,
                    "x2": 0,
                    "y2": 1,
                    "colorStops": [{
                        "offset": 0,
                        "color": "#2E86C1"
                    }, {
                        "offset": 1,
                        "color": "#1ABC9C"
                    }]
                }
            }
        }]
    }
    return options

def create_price_size_correlation(df):
    """Create price vs size scatter plot"""
    data = df[['Building Size', 'Property Price', 'Property Type']].values.tolist()
    
    options = {
        "title": {
            "text": "Property Price vs Building Size",
            "left": "center"
        },
        "tooltip": {
            "trigger": "item",
            "formatter": "function(params) { return 'Size: ' + params.value[0] + ' sq ft<br>Price: $' + params.value[1] + '<br>Type: ' + params.value[2]; }"
        },
        "xAxis": {
            "type": "value",
            "name": "Building Size (sq ft)",
            "nameLocation": "middle",
            "nameGap": 30
        },
        "yAxis": {
            "type": "value",
            "name": "Price ($)",
            "nameLocation": "middle",
            "nameGap": 30
        },
        "series": [{
            "type": "scatter",
            "data": data,
            "symbolSize": 10,
            "itemStyle": {"color": "#2E86C1"}
        }]
    }
    return options

def create_crime_rate_impact(df):
    """Create crime rate impact scatter plot"""
    if 'Crime_Rate_Percentage' in df.columns:
        data = df[['Crime_Rate_Percentage', 'Property Price', 'City']].values.tolist()
        
        options = {
            "title": {
                "text": "Property Price vs Crime Rate",
                "left": "center"
            },
            "tooltip": {
                "trigger": "item",
                "formatter": "function(params) { return 'Crime Rate: ' + params.value[0] + '%<br>Price: $' + params.value[1] + '<br>City: ' + params.value[2]; }"
            },
            "xAxis": {
                "type": "value",
                "name": "Crime Rate (%)",
                "nameLocation": "middle",
                "nameGap": 30
            },
            "yAxis": {
                "type": "value",
                "name": "Price ($)",
                "nameLocation": "middle",
                "nameGap": 30
            },
            "series": [{
                "type": "scatter",
                "data": data,
                "symbolSize": 10,
                "itemStyle": {"color": "#E74C3C"}
            }]
        }
        return options
    return None

def create_feature_correlation(df):
    """Create feature correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    data = []
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            data.append([i, j, round(corr_matrix.iloc[i, j], 2)])
    
    options = {
        "title": {
            "text": "Feature Correlation Heatmap",
            "left": "center"
        },
        "tooltip": {
            "position": "top",
            "formatter": "function(params) { return numeric_cols[params.value[0]] + ' vs ' + numeric_cols[params.value[1]] + ': ' + params.value[2]; }"
        },
        "grid": {
            "height": "50%",
            "top": "10%"
        },
        "xAxis": {
            "type": "category",
            "data": numeric_cols.tolist(),
            "splitArea": {"show": True}
        },
        "yAxis": {
            "type": "category",
            "data": numeric_cols.tolist(),
            "splitArea": {"show": True}
        },
        "visualMap": {
            "min": -1,
            "max": 1,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": "15%",
            "inRange": {
                "color": ["#E74C3C", "#FFFFFF", "#2E86C1"]
            }
        },
        "series": [{
            "type": "heatmap",
            "data": data,
            "label": {"show": True},
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowColor": "rgba(0, 0, 0, 0.5)"
                }
            }
        }]
    }
    return options

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
        st_echarts(create_price_distribution(df), height="400px")
    
    with col2:
        st.subheader("🏠 Price by Property Type")
        st_echarts(create_price_by_property_type(df), height="400px")
    
    # Second row - Price by Bedrooms
    st.subheader("🛏️ Price by Number of Bedrooms")
    st_echarts(create_price_by_bedrooms(df), height="400px")
    
    # Third row - Price by City
    st.subheader("🌆 Price by City")
    st_echarts(create_price_by_city(df), height="400px")

if __name__ == "__main__":
    main()