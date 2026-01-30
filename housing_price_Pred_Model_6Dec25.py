import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 500

# Generate features
Size_sqft = np.random.randint(500, 4000, size=n_samples)
Num_bedrooms = np.random.randint(1, 6, size=n_samples)
Num_bathrooms = np.random.randint(1, 4, size=n_samples)
Age = np.random.randint(0, 50, size=n_samples)
Location = np.random.choice(['Urban', 'Suburban', 'Rural'], size=n_samples)

# Generate target variable (Price) with some random noise
Price = (
    Size_sqft * 200 + 
    Num_bedrooms * 10000 + 
    Num_bathrooms * 5000 - 
    Age * 1000 + 
    np.where(Location == 'Urban', 50000, np.where(Location == 'Suburban', 20000, 0)) +
    np.random.randint(-10000, 10000, size=n_samples)  # noise
)

# Create DataFrame
housing_df = pd.DataFrame({
    'Size_sqft': Size_sqft,
    'Num_bedrooms': Num_bedrooms,
    'Num_bathrooms': Num_bathrooms,
    'Age': Age,
    'Location': Location,
    'Price': Price
})

# Save to CSV
housing_df.to_csv('dummy_housing_data.csv', index=False)

housing_df.head()
