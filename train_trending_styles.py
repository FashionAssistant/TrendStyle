import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data_frame_1 = pd.read_csv("./shein_mens_fashion.csv")

# Step 1: Define the Target Variable (Trending or Not)
data_frame_1['is_trending'] = np.where(
    (data_frame_1['average_rating'] > 4.0) & 
    (data_frame_1['reviews_count'] > 50) & 
    (data_frame_1['discount_percentage'] > 20),
    1, 0
)

# Step 2: Preprocess the Data
# Drop unnecessary columns
columns_to_drop = ['product_id', 'sku', 'url', 'sale_price/amount_with_symbol',
                   'retail_price/amount_with_symbol', 'description', 'title']
data = data_frame_1.drop(columns_to_drop, axis=1)

# Encode categorical columns
label_encoders = {}
categorical_columns = ['color', 'category_name']
for column in categorical_columns:
    if data[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column].astype(str))

# Scale numerical features
scaler = StandardScaler()
numerical_columns = ['sale_price/amount', 'retail_price/amount', 'discount_percentage',
                     'reviews_count', 'average_rating']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Step 3: Split the Data
X = data.drop('is_trending', axis=1)
y = data['is_trending']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Train the Model
model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Save the Model, Scaler, LabelEncoders, and Feature Names
joblib.dump(model, 'trending_styles_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')  # Save feature names

print("Model, Scaler, LabelEncoders, and Feature Names saved successfully.")
