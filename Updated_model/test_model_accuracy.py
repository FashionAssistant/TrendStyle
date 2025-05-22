import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the dataset
data_frame = pd.read_csv("./modified_file.csv")

# Define the Target Variable
data_frame['is_trending'] = np.where(
    (data_frame['average_rating'] > 4.0) & 
    (data_frame['reviews_count'] > 50) & 
    (data_frame['discount_percentage'] > 20),
    1, 0
)

# Drop unnecessary columns
columns_to_drop = ['product_id', 'sku', 'url', 'sale_price_amount_with_symbol',
                   'retail_price_amount_with_symbol', 'description', 'title']
data = data_frame.drop(columns_to_drop, axis=1)

# Load encoders
label_encoders = joblib.load('label_encoders.pkl')
categorical_columns = ['color', 'category_name']

for column in categorical_columns:
    encoder = label_encoders.get(column)
    if encoder:
        data[column] = encoder.transform(data[column].astype(str))

# Load scaler and scale numerical features
scaler = joblib.load('scaler.pkl')
numerical_columns = ['sale_price_amount', 'retail_price_amount', 'discount_percentage',
                     'reviews_count', 'average_rating']
data[numerical_columns] = scaler.transform(data[numerical_columns])

# Prepare test features (exclude label-influencing features)
X = data.drop(['is_trending', 'average_rating', 'reviews_count', 'discount_percentage'], axis=1)
y = data['is_trending']

# Split dataset the same way as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load model
model = joblib.load('trending_styles_model.pkl')

# Predict
y_pred = model.predict(X_test)

# Evaluation
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
