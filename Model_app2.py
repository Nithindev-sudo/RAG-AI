import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# -------------------------------
# Load saved model
# -------------------------------
with open(r"C:\House_Price_Prediction\rf_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Sidebar - App navigation
# -------------------------------
st.sidebar.title("🏠 House Price Prediction App")
section = st.sidebar.selectbox(
    "Navigate to:",
    ["📄 Upload Data", "📊 Visualizations", "💰 Predict Prices"]
)

# -------------------------------
# File upload
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV file with house features", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # -------------------------------
    # Section: Upload Data
    # -------------------------------
    if section == "📄 Upload Data":
        st.header("📄 Uploaded Data")
        st.dataframe(data)
        st.info("Ensure your CSV contains all required columns:\n"
                "`Area_sqft, Bedrooms, Bathrooms, AgeOfHouse, Location, HouseType, HasParking, DistanceFromCityCenter`")

    # -------------------------------
    # Section: Visualizations
    # -------------------------------
    elif section == "📊 Visualizations":
        st.header("📊 Feature Distributions")
        numeric_features = ['Area_sqft', 'Bedrooms', 'Bathrooms', 'AgeOfHouse', 'DistanceFromCityCenter']
        for col in numeric_features:
            if col in data.columns:
                fig = px.histogram(data, x=col, nbins=20, title=f"Distribution of {col}", marginal="box")
                st.plotly_chart(fig, use_container_width=True)
        
        # Optional scatter plots
        st.subheader("Scatter Plots")
        if 'Area_sqft' in data.columns and 'Bedrooms' in data.columns:
            fig = px.scatter(data, x='Area_sqft', y='Bedrooms', color='Bathrooms',
                             size='AgeOfHouse', hover_data=['Location'],
                             title="Area vs Bedrooms colored by Bathrooms")
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Section: Predict Prices
    # -------------------------------
    elif section == "💰 Predict Prices":
        st.header("💰 Predicted House Prices")
        try:
            predictions = model.predict(data)
            data['Predicted_HousePrice'] = predictions
            st.dataframe(data)
            st.success("✅ Predictions completed successfully!")
            
            # Optional: Plot predicted prices
            if 'Area_sqft' in data.columns:
                fig = px.scatter(data, x='Area_sqft', y='Predicted_HousePrice', color='Location',
                                 size='Bedrooms', hover_data=['HouseType'],
                                 title="Predicted Price vs Area")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            st.info("Make sure your CSV contains all required columns:")
            st.write(model.named_steps['preprocessor'].transformers[0][2] + 
                     model.named_steps['preprocessor'].transformers[1][2])

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
    st.info("Required columns: `Area_sqft, Bedrooms, Bathrooms, AgeOfHouse, Location, HouseType, HasParking, DistanceFromCityCenter`")
