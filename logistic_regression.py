import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

print("--- Loading the fully cleaned dataset ---")
try:
    df = pd.read_csv('speed_dating_fully_cleaned.csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'speed_dating_fully_cleaned.csv' not found. Please run the preprocessing scripts first.")
    exit()

print("\n--- Preparing data for model training ---")
feature_columns = [
    'age', 'gender', 'field_cd', 'imprace', 'imprelig',
    'goal', 'date', 'go_out', 'samerace',
    'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1',
    'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1',
    'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like'
]

y = df['match']
X = df[feature_columns]
X = pd.get_dummies(X, columns=['goal', 'date', 'go_out', 'field_cd'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Data split into training set ({X_train.shape[0]} rows) and testing set ({X_test.shape[0]} rows).")


print("\n--- Applying SMOTE to balance the training data ---")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Training set shape before SMOTE: {X_train.shape}")
print(f"Training set shape after SMOTE: {X_train_resampled.shape}")


print("\n--- Scaling the feature data ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)


print("\n--- Training the Logistic Regression model on balanced data ---")
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train_scaled, y_train_resampled)
print("Model training complete.")
print("\n--- Evaluating model performance on the test set ---")
y_pred = log_reg_model.predict(X_test_scaled)
print("\nClassification Report (After SMOTE):")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Match', 'Match'], yticklabels=['No Match', 'Match'])
plt.title('Confusion Matrix - Logistic Regression (After SMOTE)', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()


print("\n--- Analyzing feature importance based on model coefficients ---")
feature_names = X.columns
coefficients = log_reg_model.coef_[0]

coeff_series = pd.Series(coefficients, index=feature_names)
sorted_coeffs = coeff_series.sort_values()

plt.figure(figsize=(12, 10))
pd.concat([sorted_coeffs.head(15), sorted_coeffs.tail(15)]).plot(kind='barh')
plt.title('Logistic Regression Coefficient Analysis (Most Influential Features)', fontsize=16)
plt.xlabel('Coefficient Value (Impact on Match Likelihood)', fontsize=12)
plt.tight_layout()
plt.show()

print("\n--- Logistic Regression analysis with SMOTE  ---")
