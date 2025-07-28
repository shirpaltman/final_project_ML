import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Loading the fully cleaned dataset ---")
df = pd.read_csv('speed_dating_fully_cleaned.csv')

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

print("\n--- Training the XGBoost model ---")
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
xgb_model.fit(X_train_resampled, y_train_resampled)
print("Model training complete.")

print("\n--- Evaluating model performance on the test set ---")
y_pred = xgb_model.predict(X_test)
print("\nClassification Report (XGBoost with SMOTE):")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['No Match', 'Match'], yticklabels=['No Match', 'Match'])
plt.title('Confusion Matrix - XGBoost', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()

y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"AUC Score: {roc_auc:.4f}")

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve - XGBoost', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

importances = xgb_model.feature_importances_
feature_names = X_train_resampled.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette='magma')
plt.title('Top 20 Most Important Features - XGBoost', fontsize=16)
plt.xlabel('Importance Score (F-score)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()

print("\n--- XGBoost analysis complete ---")
