import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
data = pd.read_csv(r"C:\House_Price_Prediction\dummy_housing_data.csv")

# Features and target
X = data.drop('Price', axis=1)
y = data['Price']

# Identify numeric and categorical features
numeric_features = ['Size_sqft', 'Num_bedrooms', 'Num_bathrooms', 'Age']
categorical_features = ['Location']

# Preprocessing pipelines
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate models
for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name} Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")
    print('-'*30)

# Predict prices for new properties
new_properties = pd.DataFrame({
    'Size_sqft': [2000, 3000],
    'Num_bedrooms': [3, 4],
    'Num_bathrooms': [2, 3],
    'Age': [10, 5],
    'Location': ['Urban', 'Suburban']
})

# Use Random Forest pipeline for prediction (can use any model)
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
rf_pipeline.fit(X, y)
predicted_prices = rf_pipeline.predict(new_properties)
print("Predicted Prices for New Properties:", predicted_prices)
