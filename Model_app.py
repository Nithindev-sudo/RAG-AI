import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# -------------------------------
# 1. Load the saved model
# -------------------------------
with open(r"C:\House_Price_Prediction\rf_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# 2. Streamlit App
# -------------------------------
st.title("House Price Prediction Dashboard")
st.markdown("Upload a CSV file with house features to get price predictions.")

# -------------------------------
# 3. File upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read the uploaded CSV
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(data)

    # -------------------------------
    # 4. Visualizations
    # -------------------------------
    st.subheader("Feature Distributions")

    numeric_features = ['Area_sqft', 'Bedrooms', 'Bathrooms', 'AgeOfHouse', 'DistanceFromCityCenter']

    for col in numeric_features:
        if col in data.columns:
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, ax=ax, color='skyblue')
            ax.set_title(f'Distribution of {col}')
            st.pyplot(fig)

    # -------------------------------
    # 5. Prediction
    # -------------------------------
    st.subheader("Predicted House Prices")
    
    try:
        predictions = model.predict(data)
        data['Predicted_HousePrice'] = predictions
        st.dataframe(data)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        st.info("Make sure your CSV contains all required columns:")
        st.write(model.named_steps['preprocessor'].transformers[0][2] + 
                 model.named_steps['preprocessor'].transformers[1][2])

