import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def predict_trending(input_data):
    # Load the saved model, preprocessors, and feature names
    model = joblib.load('trending_styles_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')  # Load saved feature names

    # Encode categorical columns
    for column, encoder in label_encoders.items():
        if column in input_data:
            input_data[column] = encoder.transform([input_data[column]])[0]

    # Scale numerical columns
    numerical_columns = ['sale_price/amount', 'retail_price/amount', 'discount_percentage',
                         'reviews_count', 'average_rating']
    numerical_values = np.array([input_data[col] for col in numerical_columns]).reshape(1, -1)
    scaled_values = scaler.transform(numerical_values)

    # Update input data with scaled values
    for i, col in enumerate(numerical_columns):
        input_data[col] = scaled_values[0][i]

    # Convert input to DataFrame and ensure columns match training feature names
    input_df = pd.DataFrame([input_data])
    for col in feature_names:
        if col not in input_df:
            input_df[col] = 0  # Add missing columns with default value
    input_df = input_df[feature_names]  # Align column order

    # Predict
    prediction = model.predict(input_df)
    return 'Trending' if prediction[0] == 1 else 'Not Trending'


# Example usage
example_input = {
    'color': 'Black',
    'category_name': 'Men T-Shirts',
    'sale_price/amount': 15.99,
    'retail_price/amount': 25.99,
    'discount_percentage': 40,
    'reviews_count': 100,
    'average_rating': 4.8
}

# example_input = {
#     'color': 'Blue',
#     'category_name': 'Men T-Shirts',
#     'sale_price/amount': 20.00,
#     'retail_price/amount': 25.00,
#     'discount_percentage': 15,  # Low discount
#     'reviews_count': 10,       # Few reviews
#     'average_rating': 3.5     # Low rating
# }      

result = predict_trending(example_input)
print(f"Prediction: {result}")