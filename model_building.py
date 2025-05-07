# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the dataset
# Replace 'cleaned_diabetes_data_1.csv' with your actual dataset file name
df = pd.read_csv('cleaned_diabetes_data_1.csv')

# Step 3: Handle missing values
# Check if there are any missing values in the dataset
print("Missing values before imputation:\n", df.isnull().sum())

# Impute missing values with the median of each column
df = df.fillna(df.median())

# If the target variable (e.g., 'Outcome') has missing values, impute with the most frequent value (mode)
df['Outcome'] = df['Outcome'].fillna(df['Outcome'].mode()[0])

# Check if missing values are handled
print("\nMissing values after imputation:\n", df.isnull().sum())

# Step 4: Prepare features and target variable
# Assuming 'Outcome' is the target variable and other columns are features
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']  # Target variable

# Step 5: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Initialize the model (Logistic Regression)
model = LogisticRegression(max_iter=1000)

# Step 7: Train the model on the training data
model.fit(X_train, y_train)

# Step 8: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 9: Evaluate the model's performance
print("\nModel Evaluation:")

# Accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report (Precision, Recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
