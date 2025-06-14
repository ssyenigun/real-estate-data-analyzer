import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Real Estate Data Analysis - Clustering Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Real Estate Data Clustering Analysis")

def perform_clustering_analysis(df):
    """
    Perform clustering analysis on city data based on property prices and crime rates.
    
    Args:
        df (pd.DataFrame): DataFrame containing city-level summary data
    """
    # Create city-level summary
    city_summary = df.groupby('City').agg({
        'Property Price': 'mean',
        'Building Size': 'mean',
        'Bedrooms': 'mean',
        'Bathrooms': 'mean'
    }).reset_index()
    
    # Add hypothetical crime rate (for demonstration)
    # In a real scenario, you would use actual crime data
    np.random.seed(42)
    city_summary['Hypothetical Crime Percentage'] = np.random.uniform(1, 10, size=len(city_summary))
    
    # Select features for clustering
    X_cluster = city_summary[['Property Price', 'Hypothetical Crime Percentage']].copy()

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Determine the optimal number of clusters using both Elbow Method and Silhouette Score
    inertia = []
    silhouette_scores = []
    max_clusters = min(10, len(city_summary) - 1)
    
    if max_clusters > 1:
        for k in range(2, max_clusters + 1):  # Start from 2 for silhouette score
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
            
            # Calculate silhouette score
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(X_scaled, labels))

        # Create two columns for the plots
        col1, col2 = st.columns(2)
        
        # Plot the elbow curve
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(2, max_clusters + 1), inertia, marker='o')
            ax.set_title('Elbow Method for Optimal K')
            ax.set_xlabel('Number of Clusters (K)')
            ax.set_ylabel('Inertia')
            ax.set_xticks(range(2, max_clusters + 1))
            ax.grid(True)
            st.pyplot(fig)
            plt.close()
        
        # Plot silhouette scores
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
            ax.set_title('Silhouette Score Analysis')
            ax.set_xlabel('Number of Clusters (K)')
            ax.set_ylabel('Silhouette Score')
            ax.set_xticks(range(2, max_clusters + 1))
            ax.grid(True)
            st.pyplot(fig)
            plt.close()

        # Choose number of clusters
        n_clusters = st.slider(
            "Select number of clusters",
            min_value=2,
            max_value=max_clusters,
            value=min(3, max_clusters)
        )
        
        if n_clusters > 0:
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            city_summary['Cluster'] = kmeans.fit_predict(X_scaled)

            # Add scaled values for plotting
            city_summary['Property Price_scaled'] = X_scaled[:, 0]
            city_summary['Hypothetical Crime Percentage_scaled'] = X_scaled[:, 1]

            # Create two columns for the visualizations
            col1, col2 = st.columns(2)
            
            # Visualize the clusters (scaled)
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    x='Property Price_scaled',
                    y='Hypothetical Crime Percentage_scaled',
                    hue='Cluster',
                    data=city_summary,
                    palette='viridis',
                    s=100,
                    ax=ax
                )
                ax.set_title(f'City Clusters (Scaled Features)')
                ax.set_xlabel('Scaled Property Price')
                ax.set_ylabel('Scaled Crime Percentage')
                ax.grid(True)
                st.pyplot(fig)
                plt.close()
            
            # Visualize the clusters (original scale)
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    x='Property Price',
                    y='Hypothetical Crime Percentage',
                    hue='Cluster',
                    data=city_summary,
                    palette='viridis',
                    s=100,
                    ax=ax
                )
                ax.set_title(f'City Clusters (Original Scale)')
                ax.set_xlabel('Property Price')
                ax.set_ylabel('Crime Percentage')
                ax.grid(True)
                st.pyplot(fig)
                plt.close()

            # Display cluster characteristics
            st.subheader("Cluster Characteristics")
            cluster_characteristics = city_summary.groupby('Cluster').agg({
                'Property Price': ['mean', 'std'],
                'Hypothetical Crime Percentage': ['mean', 'std'],
                'Building Size': 'mean',
                'Bedrooms': 'mean',
                'Bathrooms': 'mean'
            }).round(2)
            st.dataframe(cluster_characteristics)

            # Display cities by cluster
            st.subheader("Cities by Cluster")
            for cluster_id in sorted(city_summary['Cluster'].unique()):
                cities_in_cluster = city_summary[city_summary['Cluster'] == cluster_id]['City'].tolist()
                st.write(f"**Cluster {cluster_id}:** {', '.join(cities_in_cluster)}")
            
            # Calculate and display silhouette score for the chosen number of clusters
            silhouette_avg = silhouette_score(X_scaled, city_summary['Cluster'])
            st.write(f"\n**Silhouette Score:** {silhouette_avg:.3f}")
            st.write("A higher silhouette score indicates better-defined clusters.")
            
            return city_summary, cluster_characteristics
        else:
            st.warning("Not enough data points (cities) to perform clustering with more than 1 cluster.")
            return None, None
    else:
        st.warning("Not enough data points (cities) to perform clustering.")
        return None, None

def explain_clustering_results(cluster_characteristics):
    """
    Provide an explanation of the clustering results.
    
    Args:
        cluster_characteristics (pd.DataFrame): DataFrame containing mean values for each cluster
    """
    if cluster_characteristics is None:
        return
    
    st.subheader("Clustering Analysis Explanation")
    st.write("The cities have been grouped into clusters based on their property prices and crime rates.")
    st.write("Each cluster represents a distinct group of cities with similar characteristics:")
    
    for cluster_id in cluster_characteristics.index:
        price = cluster_characteristics.loc[cluster_id, ('Property Price', 'mean')]
        price_std = cluster_characteristics.loc[cluster_id, ('Property Price', 'std')]
        crime = cluster_characteristics.loc[cluster_id, ('Hypothetical Crime Percentage', 'mean')]
        crime_std = cluster_characteristics.loc[cluster_id, ('Hypothetical Crime Percentage', 'std')]
        
        st.write(f"\n**Cluster {cluster_id}:**")
        st.write(f"- Average Property Price: ${price:,.2f} (±${price_std:,.2f})")
        st.write(f"- Average Crime Rate: {crime:.2f}% (±{crime_std:.2f}%)")
        
        # Provide interpretation based on values
        if price > cluster_characteristics[('Property Price', 'mean')].mean():
            price_desc = "higher than average"
        else:
            price_desc = "lower than average"
            
        if crime > cluster_characteristics[('Hypothetical Crime Percentage', 'mean')].mean():
            crime_desc = "higher than average"
        else:
            crime_desc = "lower than average"
            
        st.write(f"This cluster represents areas with {price_desc} property prices and {crime_desc} crime rates.")
        
        # Add interpretation of standard deviations
        st.write(f"The standard deviation of property prices (${price_std:,.2f}) and crime rates ({crime_std:.2f}%) indicates the variability within this cluster.")

# Main execution
if 'processed_data' in st.session_state:
    df = st.session_state.processed_data
    
    # Perform clustering analysis
    city_summary, cluster_characteristics = perform_clustering_analysis(df)
    
    # Explain results
    if cluster_characteristics is not None:
        explain_clustering_results(cluster_characteristics)
else:
    st.error("No processed data found. Please upload and process your data first.") 