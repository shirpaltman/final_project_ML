import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Loading the fully cleaned dataset ---")
try:
    df = pd.read_csv('speed_dating_fully_cleaned.csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'speed_dating_fully_cleaned.csv' not found.")
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
X = pd.get_dummies(df[feature_columns], columns=['goal', 'date', 'go_out', 'field_cd'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("\n--- Applying SMOTE to balance the training data ---")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n--- Scaling the feature data ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

print("\n--- Training the SVM model ---")
svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train_resampled)
print("Model training complete.")

print("\n--- Evaluating model performance on the test set ---")
y_pred = svm_model.predict(X_test_scaled)
print("\nClassification Report (SVM with SMOTE):")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Match', 'Match'], yticklabels=['No Match', 'Match'])
plt.title('Confusion Matrix - SVM', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()

print("\n--- Generating Decision Boundary Visualization ---")


pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)


svm_plot_model = SVC(kernel='rbf', C=1.0, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
svm_plot_model.fit(X_train_pca, y_train_resampled)


x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))


Z = svm_plot_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure(figsize=(10, 8))

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
plt.title('SVM Decision Boundary with PCA-Reduced Data', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(handles=scatter.legend_elements()[0], labels=['No Match', 'Match'])
plt.show()
