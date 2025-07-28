import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('speed_dating_fully_cleaned.csv')
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

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred = rf_model.predict(X_test)

print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

importances = rf_model.feature_importances_
feature_names = X_train_resampled.columns
feature_importance_series = pd.Series(importances, index=feature_names)

plt.figure(figsize=(10, 8))
feature_importance_series.nlargest(20).sort_values().plot(kind='barh')
plt.title('Top 20 Most Important Features - Random Forest')
plt.xlabel('Gini Importance')
plt.show()