import pandas as pd


try:
    df = pd.read_csv('speed_dating_partially_cleaned.csv')
    print("Loaded  Shape:", df.shape)
except FileNotFoundError:
    print("Missing file")
    exit()

print("\nImputing missing values")
df_clean = df.copy()

for col in df_clean.columns:
    if df_clean[col].isnull().any():
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif df_clean[col].dtype == 'object':
            mode = df_clean[col].mode()
            if not mode.empty:
                df_clean[col] = df_clean[col].fillna(mode[0])

if df_clean.isnull().any().any():
    print("Still has some NaNs.")
else:
    print("All NaNs filled.")

print("Final shape:", df_clean.shape)
df_clean.to_csv('speed_dating_fully_cleaned.csv', index=False)
print("Saved.")
