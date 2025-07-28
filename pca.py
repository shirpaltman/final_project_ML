import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Loading and preparing the data ---")
df = pd.read_csv('speed_dating_fully_cleaned.csv')

feature_columns = [
    'age', 'gender', 'field_cd', 'imprace', 'imprelig',
    'goal', 'date', 'go_out', 'samerace',
    'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1',
    'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1',
    'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'like'
]
y = df['match']
X = pd.get_dummies(df[feature_columns], columns=['goal', 'date', 'go_out', 'field_cd'], drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data preparation complete.")

print("\n--- Fitting PCA and analyzing components ---")
pca = PCA(n_components=2)
pca.fit(X_scaled)

loadings = pca.components_
loadings_df = pd.DataFrame(loadings.T, columns=['PC1', 'PC2'], index=X.columns)
print("\n--- Top 10 Features for Principal Component 1 ---")
pc1_top_features = loadings_df['PC1'].abs().sort_values(ascending=False).head(10)
print(pc1_top_features)
print("\n--- Top 10 Features for Principal Component 2 ---")
pc2_top_features = loadings_df['PC2'].abs().sort_values(ascending=False).head(10)
print(pc2_top_features)

plt.figure(figsize=(8, 12))
sns.heatmap(loadings_df, annot=True, cmap='viridis', fmt='.2f')
plt.title('PCA Component Loadings for Original Features', fontsize=16)
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Original Feature', fontsize=12)
plt.show()

print("\n Analysis Complete ")
