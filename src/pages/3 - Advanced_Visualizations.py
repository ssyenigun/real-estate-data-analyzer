import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def create_monthly_avg_price(df):
    """Create monthly average property price visualization with rolling average"""
    if 'Published Date' not in df.columns or 'Property Price' not in df.columns:
        return None
        
    df_copy = df.copy() # Avoid SettingWithCopyWarning
    # Convert dates and extract month
    df_copy['Published Date'] = pd.to_datetime(df_copy['Published Date'])
    df_copy['Month'] = df_copy['Published Date'].dt.to_period('M')
    
    # Calculate monthly averages
    monthly_avg = df_copy.groupby('Month')['Property Price'].mean().reset_index()
    monthly_avg['Month'] = monthly_avg['Month'].astype(str)
    
    # Calculate 3-month rolling average
    monthly_avg['Rolling_Avg'] = monthly_avg['Property Price'].rolling(window=3, min_periods=1).mean()
    
    option = {
        "title": {
            "text": "Monthly Average Property Price Over Time with Rolling Average",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
        },
        "legend": { # Add legend for multiple series
            "data": ["Monthly Average Price", "3-Month Rolling Average"],
            "bottom": 0
        },
        "xAxis": {
            "type": "category",
            "data": monthly_avg['Month'].tolist(),
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {
            "type": "value",
            "name": "Average Price ($)",
            "axisLabel": {"formatter": "${value}"}
        },
        "series": [{
            "name": "Monthly Average Price",
            "data": monthly_avg['Property Price'].tolist(),
            "type": "line",
            "smooth": True,
            "itemStyle": {"color": "#2E86C1"},
            "areaStyle": {"opacity": 0.3}
        },
        {
            "name": "3-Month Rolling Average",
            "data": monthly_avg['Rolling_Avg'].tolist(),
            "type": "line",
            "smooth": True,
            "itemStyle": {"color": "#E74C3C"},
            "lineStyle": {"width": 3}
        }],
        "grid": {"bottom": "15%"}
    }
    return option

def create_crime_rate_vs_price(df):
    """Create crime rate vs property price visualization"""
    if 'Crime_Rate_Percentage' not in df.columns or 'Property Price' not in df.columns:
        return None
        
    option = {
        "title": {
            "text": "Crime Rate vs Property Price",
            "left": "center"
        },
        "tooltip": {
            "trigger": "item",
        },
        "xAxis": {
            "type": "value",
            "name": "Crime Rate (%)",
            "nameLocation": "middle",
            "nameGap": 30
        },
        "yAxis": {
            "type": "value",
            "name": "Property Price ($)",
            "nameLocation": "middle",
            "nameGap": 70,
            "axisLabel": {"formatter": "${value}"}
        },
        "series": [{
            "type": "scatter",
            "data": df[['Crime_Rate_Percentage', 'Property Price']].values.tolist(),
            "symbolSize": 8,
            "itemStyle": {"color": "#E74C3C"}
        }]
    }
    return option



def create_bedroom_vs_price(df):
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

def create_bathroom_vs_price(df):
    """Create bathroom count vs property price visualization"""
    if 'Bathrooms' not in df.columns or 'Property Price' not in df.columns:
        return None
        
    df_cleaned = df.dropna(subset=['Bathrooms', 'Property Price']).copy()
    if df_cleaned.empty:
        return None

    df_cleaned['Bathrooms'] = pd.to_numeric(df_cleaned['Bathrooms'], errors='coerce')
    df_cleaned = df_cleaned.dropna(subset=['Bathrooms'])

    bathroom_avg = df_cleaned.groupby('Bathrooms')['Property Price'].agg(['mean', 'count']).reset_index()
    bathroom_avg = bathroom_avg[bathroom_avg['count'] >= 1]
    
    bathroom_avg['Bathrooms'] = bathroom_avg['Bathrooms'].astype(int)
    bathroom_avg = bathroom_avg.sort_values('Bathrooms')

    option = {
        "title": {
            "text": "Average Property Price by Number of Bathrooms",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
        },
        "xAxis": {
            "type": "category",
            "data": bathroom_avg['Bathrooms'].tolist(),
            "name": "Number of Bathrooms",
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
            "data": bathroom_avg['mean'].tolist(),
            "type": "bar",
            "itemStyle": {"color": "#3498DB"}
        }]
    }
    return option

def create_property_type_vs_price(df):
    """Create property type vs price visualization"""
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

def create_top_10_expensive_cities(df):
    """Create top 10 most expensive cities visualization"""
    if 'City' not in df.columns or 'Property Price' not in df.columns:
        return None
        
    city_avg = df.groupby('City')['Property Price'].mean().sort_values(ascending=False).head(10)
    
    option = {
        "title": {
            "text": "Top 10 Most Expensive Cities",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
       
        },
        "xAxis": {
            "type": "category",
            "data": city_avg.index.tolist(),
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
            "data": city_avg.values.tolist(),
            "type": "bar",
            "itemStyle": {"color": "#E67E22"}
        }],
        "grid": {"bottom": "15%"}
    }
    return option

def create_top_10_avg_sqft_cities(df):
    """Create top 10 cities by average square footage visualization"""
    if 'City' not in df.columns or 'Building Size' not in df.columns:
        return None
        
    city_avg = df.groupby('City')['Building Size'].mean().sort_values(ascending=False).head(10)
    
    option = {
        "title": {
            "text": "Top 10 Cities by Average Square Footage",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
        },
        "xAxis": {
            "type": "category",
            "data": city_avg.index.tolist(),
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {
            "type": "value",
            "name": "Average Size (sq ft)",
            "nameLocation": "middle",
            "nameGap": 70,
            "axisLabel": {"formatter": "{value}"}
        },
        "series": [{
            "data": city_avg.values.tolist(),
            "type": "bar",
            "itemStyle": {"color": "#16A085"}
        }],
        "grid": {"bottom": "15%"}
    }
    return option

def create_price_per_sqft_by_city(df):
    """Create visualization for average property price per square foot by city (Top 10)"""
    if 'Property Price' not in df.columns or 'Building Size' not in df.columns or 'City' not in df.columns:
        return None
    
    df_filtered = df[(df['Building Size'] > 0) & (df['Property Price'].notna())].copy()
    if df_filtered.empty:
        return None

    df_filtered['Price_per_sqft'] = df_filtered['Property Price'] / df_filtered['Building Size']
    
    city_avg_prices = df_filtered.groupby('City')['Price_per_sqft'].mean().sort_values(ascending=False).head(10)

    option = {
        "title": {
            "text": "Top 10 Cities by Average Price per Square Foot",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
            "formatter": "City: {b}<br/>Avg Price/SqFt: ${c:,.2f}"
        },
        "xAxis": {
            "type": "category",
            "data": city_avg_prices.index.tolist(),
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {
            "type": "value",
            "name": "Average Price ($/SqFt)",
            "axisLabel": {"formatter": "${value}"}
        },
        "series": [{
            "data": city_avg_prices.values.tolist(),
            "type": "bar",
            "itemStyle": {"color": "#8E44AD"}
        }],
        "grid": {"bottom": "15%"}
    }
    return option

def create_city_price_top_10_others(df):
    """Create visualization for average property price by city (Top 10 and Others)"""
    if 'City' not in df.columns or 'Property Price' not in df.columns:
        return None
    
    df_cleaned = df.dropna(subset=['City', 'Property Price']).copy()
    if df_cleaned.empty:
        return None

    top_10_cities_names = df_cleaned['City'].value_counts().nlargest(10).index.tolist()
    
    df_top_10 = df_cleaned[df_cleaned['City'].isin(top_10_cities_names)]
    df_others = df_cleaned[~df_cleaned['City'].isin(top_10_cities_names)]

    avg_prices_top_10 = df_top_10.groupby('City')['Property Price'].mean().sort_values(ascending=False)
    avg_price_others = df_others['Property Price'].mean() if not df_others.empty else 0

    option = {
        "title": {
            "text": "Average Property Price by City (Top 10 + Others)",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
        },
        "xAxis": {
            "type": "category",
            "data": avg_prices_top_10.index.tolist(),
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {
            "type": "value",
            "name": "Average Property Price ($)",
            "axisLabel": {"formatter": "${value}"}
        },
        "series": [{
            "name": "Average Price",
            "data": avg_prices_top_10.values.tolist(),
            "type": "bar",
            "itemStyle": {"color": "#AF7AC5"},
            "markLine": {
                "silent": True,
                "data": [
                    {
                        "yAxis": avg_price_others,
                        "name": "Other Cities Average",
                        "lineStyle": {"type": "dashed", "color": "#F39C12"},
                    }
                ] if avg_price_others > 0 else []
            }
        }],
        "grid": {"bottom": "20%"}
    }
    return option

def create_elbow_plot(df):
    """Create elbow plot for K-means clustering"""
    if 'Numeric_Price' not in df.columns or 'Total_Crime_Rate' not in df.columns:
        return None
        
    # Prepare data for clustering
    X = df[['Numeric_Price', 'Total_Crime_Rate']].dropna()
    if X.empty: # Check if X is empty after dropping NaNs
        return None
    X_scaled = StandardScaler().fit_transform(X)
    
    # Calculate inertia for different k values
    inertias = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Added n_init for KMeans
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Create elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.xticks(K)
    
    return plt

def create_cluster_scatter(df):
    """Create scatter plot of city clusters"""
    if 'Numeric_Price' not in df.columns or 'Total_Crime_Rate' not in df.columns or 'City' not in df.columns:
        return None
        
    # Prepare data for clustering
    X = df[['Numeric_Price', 'Total_Crime_Rate']].dropna()
    if X.empty: # Check if X is empty after dropping NaNs
        return None
    X_scaled = StandardScaler().fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # Added n_init for KMeans
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('Scaled Price')
    plt.ylabel('Scaled Crime Rate')
    plt.title('City Clusters (K=3)')
    plt.colorbar(scatter, label='Cluster')
    
    return plt

def create_cluster_summary(df):
    """Create summary table of clusters"""
    if 'Numeric_Price' not in df.columns or 'Total_Crime_Rate' not in df.columns or 'City' not in df.columns:
        return None
        
    # Prepare data for clustering
    X = df[['Numeric_Price', 'Total_Crime_Rate']].dropna()
    if X.empty: # Check if X is empty after dropping NaNs
        return None
    X_scaled = StandardScaler().fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # Added n_init for KMeans
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create summary DataFrame
    cluster_df = pd.DataFrame({
        'City': df['City'].iloc[X.index],
        'Cluster': clusters,
        'Price': X['Numeric_Price'],
        'Crime_Rate': X['Total_Crime_Rate']
    })
    
    # Calculate cluster statistics
    cluster_stats = cluster_df.groupby('Cluster').agg({
        'Price': ['mean', 'median', 'count'],
        'Crime_Rate': ['mean', 'median']
    }).round(2)
    
    return cluster_stats

def create_feature_importance(df):
    """Create feature importance plot using Random Forest"""
    if 'Numeric_Price' not in df.columns:
        return None
        
    # Prepare features
    features = ['Numeric_Size', 'Bedrooms', 'Total_Crime_Rate']
    
    # Drop rows with NaN in features or target
    df_cleaned = df.dropna(subset=features + ['Numeric_Price']).copy()
    if df_cleaned.empty:
        return None

    X = df_cleaned[features]
    y = df_cleaned['Numeric_Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance)
    plt.title('Feature Importance for Property Price Prediction')
    plt.xlabel('Importance Score')
    
    return plt

def main():
    st.title("📊 Advanced Property Analysis")
    
    # Load data
    try:
        df = pd.read_csv('src/processed_data.csv')
        
        # --- Data Cleaning and Preprocessing ---
        # Convert 'Property Price' to numeric
        if 'Property Price' in df.columns:
            df['Property Price'] = pd.to_numeric(df['Property Price'].astype(str).str.replace(',', ''), errors='coerce')
            df['Numeric_Price'] = df['Property Price'] # Create Numeric_Price for advanced analysis

        # Clean and convert 'Building Size' to numeric
        if 'Building Size' in df.columns:
            df['Building Size'] = df['Building Size'].astype(str).str.replace(',', '').str.replace(' sq ft', '').str.strip()
            df['Building Size'] = pd.to_numeric(df['Building Size'], errors='coerce')
            df['Numeric_Size'] = df['Building Size'] # Create Numeric_Size for advanced analysis

        # Convert 'Crime_Rate_Percentage' to numeric and create 'Total_Crime_Rate'
        if 'Crime_Rate_Percentage' in df.columns:
            df['Crime_Rate_Percentage'] = pd.to_numeric(df['Crime_Rate_Percentage'], errors='coerce')
            df['Total_Crime_Rate'] = df['Crime_Rate_Percentage'] # Create Total_Crime_Rate for advanced analysis

        # Ensure 'Published Date' is in datetime format
        if 'Published Date' in df.columns:
            df['Published Date'] = pd.to_datetime(df['Published Date'], errors='coerce')

        # Ensure 'Bedrooms' and 'Bathrooms' are numeric
        if 'Bedrooms' in df.columns:
            df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
        if 'Bathrooms' in df.columns:
            df['Bathrooms'] = pd.to_numeric(df['Bathrooms'], errors='coerce')

        # Drop rows where essential columns are NaN after conversion
        df.dropna(subset=['Property Price', 'Building Size', 'Bedrooms', 'Bathrooms', 'Crime_Rate_Percentage', 'Published Date', 'City', 'Property Type'], inplace=True)

        # Display data info for debugging
        st.write("Data Shape after cleaning:", df.shape)
        st.write("Available Columns after cleaning:", df.columns.tolist())
        st.write("Data Head after cleaning:", df.head())
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please make sure the data file 'src/src/processed_data.csv' exists and contains the required columns.")
        return
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3 = st.tabs(["📈 Price Analysis", "🏠 Property Features", "🔍 Advanced Analysis"])
    
    with tab1:
        st.header("Price Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly Average Price (now includes rolling average)
            st.subheader("Monthly Average Property Price Over Time")
            monthly_option = create_monthly_avg_price(df)
            if monthly_option:
                st_echarts(monthly_option, height="400px")
            else:
                st.warning("Monthly price data (Published Date, Property Price) not available or insufficient.")
            
            # Crime Rate vs Price
            st.subheader("Crime Rate vs Property Price")
            crime_option = create_crime_rate_vs_price(df)
            if crime_option:
                st_echarts(crime_option, height="400px")
            else:
                st.warning("Crime rate data (Crime_Rate_Percentage, Property Price) not available or insufficient.")
        
        with col2:
            # Top 10 Expensive Cities (Existing)
            st.subheader("Top 10 Most Expensive Cities")
            expensive_option = create_top_10_expensive_cities(df)
            if expensive_option:
                st_echarts(expensive_option, height="400px")
            else:
                st.warning("City price data (City, Property Price) not available or insufficient.")
            
            # New: City Price Top 10 + Others
            st.subheader("Average Property Price by City (Top 10 + Others)")
            city_price_others_option = create_city_price_top_10_others(df)
            if city_price_others_option:
                st_echarts(city_price_others_option, height="400px")
            else:
                st.warning("City price data (City, Property Price) not available or insufficient for 'Top 10 + Others' chart.")

    with tab2:
        st.header("Property Features")
        
        col1, col2 = st.columns(2)
        
        with col1:

            
            # Bedroom vs Price (Existing)
            st.subheader("Bedroom Count vs Property Price")
            bedroom_option = create_bedroom_vs_price(df)
            if bedroom_option:
                st_echarts(bedroom_option, height="400px")
            else:
                st.warning("Bedroom data (Bedrooms, Property Price) not available or insufficient.")
            
            # New: Bathroom vs Price
            st.subheader("Bathroom Count vs Property Price")
            bathroom_option = create_bathroom_vs_price(df)
            if bathroom_option:
                st_echarts(bathroom_option, height="400px")
            else:
                st.warning("Bathroom data (Bathrooms, Property Price) not available or insufficient.")

        with col2:
            # Property Type vs Price (Existing)
            st.subheader("Property Type vs Price")
            type_option = create_property_type_vs_price(df)
            if type_option:
                st_echarts(type_option, height="400px")
            else:
                st.warning("Property type data (Property Type, Property Price) not available or insufficient.")
            
            # Top 10 Average Square Footage Cities (Existing)
            st.subheader("Top 10 Cities by Average Square Footage")
            sqft_option = create_top_10_avg_sqft_cities(df)
            if sqft_option:
                st_echarts(sqft_option, height="400px")
            else:
                st.warning("Square footage data (City, Building Size) not available or insufficient.")
            
            # New: Price per Sq Ft by City
            st.subheader("Top 10 Cities by Average Price per Sq Ft")
            price_sqft_option = create_price_per_sqft_by_city(df)
            if price_sqft_option:
                st_echarts(price_sqft_option, height="400px")
            else:
                st.warning("Price per sq ft data (Property Price, Building Size, City) not available or insufficient.")

    with tab3:
        st.header("Advanced Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Elbow Plot (Existing)
            st.subheader("K-means Clustering Analysis: Elbow Plot")
            elbow_plot = create_elbow_plot(df)
            if elbow_plot:
                st.pyplot(elbow_plot)
                plt.close()
            else:
                st.warning("Clustering data (Numeric_Price, Total_Crime_Rate) not available or insufficient for Elbow Plot.")
        
        with col2:
            # Cluster Scatter Plot (Existing)
            st.subheader("K-means Clustering Analysis: City Clusters Scatter Plot")
            cluster_scatter = create_cluster_scatter(df)
            if cluster_scatter:
                st.pyplot(cluster_scatter)
                plt.close()
            else:
                st.warning("Clustering data (Numeric_Price, Total_Crime_Rate, City) not available or insufficient for Cluster Scatter Plot.")

            # Feature Importance (Existing)
            st.subheader("Feature Importance Analysis")
            feature_importance = create_feature_importance(df)
            if feature_importance:
                st.pyplot(feature_importance)
                plt.close()
            else:
                st.warning("Feature importance data (Numeric_Size, Bedrooms, Total_Crime_Rate, Numeric_Price) not available or insufficient.")
        
        # Cluster Summary (Existing)
        cluster_stats = create_cluster_summary(df)
        if cluster_stats is not None:
            st.subheader("Cluster Summary")
            st.dataframe(cluster_stats)
            st.markdown("""
            **Cluster Interpretation:**
            - Cluster 0: High-price, low-crime areas
            - Cluster 1: Low-price, high-crime areas
            - Cluster 2: Moderate-price, moderate-crime areas
            """)
        else:
            st.warning("Cluster summary data (Numeric_Price, Total_Crime_Rate, City) not available or insufficient.")


if __name__ == "__main__":
    main() 