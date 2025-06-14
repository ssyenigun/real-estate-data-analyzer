import streamlit as st

def show_conclusion():
    st.title("Conclusion")
    
    # Investment and Development Section
    st.header("Investment and Development of Real Estate")
    st.markdown("""
    **For Luxury/Premium Development:**
    - Focus on cities in Cluster 0 (High Price, Low Crime)
    - These cities likely have a market for high-end properties
    - Security is of utmost importance to residents

    **For Affordable Housing/Redevelopment:**
    - Consider cities in Cluster 1 (Low Price, High Crime)
    - While possibly riskier, these cities may have lower entry points
    - Potential for value-integrate prospects
    - Possible assistance from revitalization programs in urban areas

    **For Balanced Development:**
    - Urban areas in Cluster 2 exhibit modest degrees of both crime and pricing
    - Good investment options that can appeal to a wide demographic
    """)

    # Urban Planning Section
    st.header("Urban Planning and Public Policy")
    st.markdown("""
    **Cluster 1 (Low Price, High Crime):**
    - Policy interventions must focus on:
        - Enhancing public safety
        - Improving community engagement activities
        - Enticing businesses to spur local economy
        - Investing in infrastructural development

    **Cluster 0 (High Price, Low Crime):**
    - Planning should focus on:
        - Managing growth
        - Maintaining infrastructure quality
        - Preserving area character
        - Addressing housing unaffordability

    **Cluster 2 (Moderate):**
    - Priority areas:
        - Promoting balanced development
        - Investing in public facilities
        - Implementing strategies to prevent decline
        - Adjusting to shifting demographic trends
    """)

    # Commercial Strategy Section
    st.header("Commercial and Marketing Strategy")
    st.markdown("""
    **High-End Goods/Services:**
    - Target Cluster 0 cities
    - Focus on advertising and store locations

    **Value/Affordability or Security Products:**
    - Target Cluster 1 urban areas
    - Potential for increased profitability

    **General Services and Retailers:**
    - Target Cluster 2 urban areas
    - Suited for standard business models
    """)

    # Residential Procurement Section
    st.header("Protocols of Residential Procurement and Relocation")
    st.markdown("""
    **Security-Conscious Buyers:**
    - Target Cluster 0 metropolitan areas
    - Higher cost but better security

    **Affordability-Focused Buyers:**
    - Consider Cluster 1 cities
    - Be prepared for potentially higher crime rates

    **Balanced Approach Buyers:**
    - Look at Cluster 2 urban locations
    - Compromise between affordability and protection
    """)

    # Results and Discussion Section
    st.header("Results and Discussion")
    st.markdown("""
    **Key Findings:**
    - Building Size, number of Bathrooms, and number of Bedrooms are the most crucial predictors of rent prices
    - Toronto was the most expensive city on average, while Hamilton was the least
    - Random Forest model produced the best prediction accuracy (lowest error, highest R²)
    - Clustering analysis revealed distinct patterns for different city groups

    **Methodological Insights:**
    - Data cleaning and feature engineering steps enabled effective modeling
    - Random Forest model outperformed others, suggesting non-linear interactions
    - Clustering of cities by price and crime rate offered actionable insights
    - Framework can guide future projects and real estate decision-making
    """)

    # References Section
    st.header("References")
    st.markdown("""
    1. Canada Crime Report. (n.d.). Crime Severity Index. Retrieved from https://canadacrimereport.com/crime-severity-index
    2. GeeksforGeeks. (n.d.). House Price Prediction using Machine Learning in Python. Retrieved from https://www.geeksforgeeks.org/house-price-prediction-using-machine-learning-in-python/
    3. Kaggle. (n.d.). Exploratory Data Analysis – House Rent Prediction [Code notebook]. Retrieved from https://www.kaggle.com/code/rkb0023/exploratory-data-analysis-house-rent-prediction
    4. Senthilkumar, V. (2023). Enhancing House Rental Price Prediction Models for the Swedish Market: Exploring External Features, Prediction Intervals and Uncertainty Management in Predicting House Rental Prices (master's thesis, KTH Royal Institute of Technology).
    5. Realtor.com. (n.d.). Ontario Rent Listings [Dataset]. Retrieved from https://www.realtor.com/international/ca/ontario/rent/p1
    """)

if __name__ == "__main__":
    show_conclusion()
