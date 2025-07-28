import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


df = pd.read_csv('speed_dating_fully_cleaned.csv')
preference_features = [
    'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1'
]
X_cluster = df[preference_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)


cluster_profile = df.groupby('cluster')[preference_features].mean()
print("Cluster Profiles (Average Preference Scores):")
print(cluster_profile)
cluster_profile.plot(kind='bar', figsize=(12, 7))
plt.title('Average Preference Score by Cluster')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.show()

df.to_csv('speed_dating_with_clusters.csv', index=False)
print("\nData with cluster assignments saved to 'speed_dating_with_clusters.csv'")
joblib.dump(kmeans, 'kmeans_model.joblib')