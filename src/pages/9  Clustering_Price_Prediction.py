import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Real Estate Data Analysis - Price Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Property Price Prediction Analysis")

def create_price_categories(df, n_categories=5):
    """
    Create price categories using quantiles.
    
    Args:
        df (pd.DataFrame): DataFrame containing property data
        n_categories (int): Number of price categories to create
    
    Returns:
        pd.Series: Series containing price categories
    """
    # Create price categories using quantiles
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    df['Price_Category'] = pd.qcut(
        df['Property Price'],
        q=n_categories,
        labels=labels
    )
    return df['Price_Category']

def prepare_features(df):
    """
    Prepare features for the model.
    
    Args:
        df (pd.DataFrame): DataFrame containing property data
    
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target variable
    """
    # Select features
    features = ['Building Size', 'Bedrooms', 'Bathrooms']
    
    # Add property type as one-hot encoded features
    property_type_dummies = pd.get_dummies(df['Property Type'], prefix='Property_Type')
    features.extend(property_type_dummies.columns)
    
    # Add city as one-hot encoded features
    city_dummies = pd.get_dummies(df['City'], prefix='City')
    features.extend(city_dummies.columns)
    
    # Create feature matrix
    X = pd.concat([
        df[['Building Size', 'Bedrooms', 'Bathrooms']],
        property_type_dummies,
        city_dummies
    ], axis=1)
    
    # Scale numerical features
    scaler = StandardScaler()
    X[['Building Size', 'Bedrooms', 'Bathrooms']] = scaler.fit_transform(
        X[['Building Size', 'Bedrooms', 'Bathrooms']]
    )
    
    return X, df['Price_Category']

def train_and_evaluate_model(X, y):
    """
    Train and evaluate the Naive Bayes model.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
    
    Returns:
        tuple: (model, X_test, y_test, y_pred) containing the trained model and test data
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot the confusion matrix.
    
    Args:
        y_test (pd.Series): True labels
        y_pred (np.array): Predicted labels
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=sorted(y_test.unique()),
        yticklabels=sorted(y_test.unique())
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Price Category')
    plt.ylabel('True Price Category')
    st.pyplot(plt)
    plt.close()

def plot_feature_importance(model, X):
    """
    Plot feature importance based on class probabilities.
    
    Args:
        model (GaussianNB): Trained Naive Bayes model
        X (pd.DataFrame): Feature matrix
    """
    # Calculate feature importance based on class probabilities
    feature_importance = np.abs(model.theta_)
    feature_importance = np.mean(feature_importance, axis=0)
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    st.pyplot(plt)
    plt.close()

def plot_price_distribution(df):
    """
    Plot the distribution of price categories.
    
    Args:
        df (pd.DataFrame): DataFrame containing price categories
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Price_Category', order=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    plt.title('Distribution of Price Categories')
    plt.xlabel('Price Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.close()

def main():
    if 'processed_data' not in st.session_state:
        st.error("No processed data found. Please upload and process your data first.")
        return
    
    df = st.session_state.processed_data
    
    # Create price categories
    st.subheader("Price Categories")
    df['Price_Category'] = create_price_categories(df)
    
    # Display price category distribution
    st.write("Price Category Distribution:")
    plot_price_distribution(df)
    
    # Show price ranges for each category
    st.write("\nPrice Ranges for Each Category:")
    price_ranges = df.groupby('Price_Category')['Property Price'].agg(['min', 'max']).round(2)
    st.dataframe(price_ranges)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Train and evaluate model
    model, X_test, y_test, y_pred = train_and_evaluate_model(X, y)
    
    # Display model performance
    st.subheader("Model Performance")
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2%}")
    
    # Classification report
    st.write("\nDetailed Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    
    # Plot confusion matrix
    st.subheader("Confusion Matrix")
    plot_confusion_matrix(y_test, y_pred)
    
    # Plot feature importance
    st.subheader("Feature Importance")
    plot_feature_importance(model, X)
    
    # Interactive prediction
    st.subheader("Make Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        building_size = st.number_input("Building Size (sq ft)", min_value=0.0)
        bedrooms = st.number_input("Number of Bedrooms", min_value=0)
        bathrooms = st.number_input("Number of Bathrooms", min_value=0)
    
    with col2:
        property_type = st.selectbox("Property Type", df['Property Type'].unique())
        city = st.selectbox("City", df['City'].unique())
    
    if st.button("Predict Price Category"):
        # Create input data
        input_data = pd.DataFrame({
            'Building Size': [building_size],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms]
        })
        
        # Add property type and city dummies
        for prop_type in df['Property Type'].unique():
            input_data[f'Property_Type_{prop_type}'] = 1 if prop_type == property_type else 0
        
        for c in df['City'].unique():
            input_data[f'City_{c}'] = 1 if c == city else 0
        
        # Ensure all features are present
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Scale numerical features
        scaler = StandardScaler()
        input_data[['Building Size', 'Bedrooms', 'Bathrooms']] = scaler.fit_transform(
            input_data[['Building Size', 'Bedrooms', 'Bathrooms']]
        )
        
        # Make prediction
        prediction = model.predict(input_data[X.columns])[0]
        
        # Display prediction
        st.success(f"Predicted Price Category: {prediction}")
        
        # Show price range for the predicted category
        category_range = df[df['Price_Category'] == prediction]['Property Price'].agg(['min', 'max'])
        st.write(f"Estimated Price Range: ${category_range['min']:,.2f} - ${category_range['max']:,.2f}")

if __name__ == "__main__":
    main() 