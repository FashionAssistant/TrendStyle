import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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

# Step 5: Predict on the Test Data
y_pred = model.predict(X_test)

# Step 6: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 7: Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Step 8: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Trending', 'Trending'], yticklabels=['Not Trending', 'Trending'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 9: Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),  # Vary the size of the training set
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all processors
    scoring='accuracy'
)

# Calculate the mean and standard deviation of the training and test scores
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_std = test_scores.std(axis=1)

# Plot the learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue')
plt.plot(train_sizes, test_mean, label='Validation Accuracy', color='green')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='green', alpha=0.2)
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Save the Model, Scaler, LabelEncoders, and Feature Names
joblib.dump(model, 'trending_styles_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')  # Save feature names

print("Model, Scaler, LabelEncoders, and Feature Names saved successfully.")
