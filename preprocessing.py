import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Heiti TC'
print(" טוען את מאגר הנתונים הגולמי ")
try:
    df_raw = pd.read_csv('Speed Dating Data.csv', encoding='ISO-8859-1')
    print(f"הנתונים נטענו בהצלחה. צורת הטבלה המקורית: {df_raw.shape}")
except FileNotFoundError:
    print("שגיאה: הקובץ 'Speed Dating Data.csv' לא נמצא.")
    exit()
print("\n--- מנתח ערכים חסרים בכל עמודה ---")
missing_percentage = df_raw.isnull().sum() * 100 / len(df_raw)
missing_percentage_sorted = missing_percentage.sort_values(ascending=False)
print("20 העמודות עם אחוז הערכים החסרים הגבוה ביותר:")
print(missing_percentage_sorted.head(20))

plt.figure(figsize=(12, 18))
missing_to_plot = missing_percentage_sorted[missing_percentage_sorted > 30]
sns.barplot(x=missing_to_plot.values, y=missing_to_plot.index, palette='viridis')
plt.title('אחוז הערכים החסרים בעמודות (מעל 30%)', fontsize=16)
plt.xlabel('אחוז ערכים חסרים (%)', fontsize=12)
plt.ylabel('שם העמודה', fontsize=12)
plt.tight_layout()
plt.show()

missing_data_threshold = 60.0
print(f"\n--- מסיר עמודות עם יותר מ-{missing_data_threshold}% ערכים חסרים ---")

columns_to_drop = missing_percentage_sorted[missing_percentage_sorted > missing_data_threshold].index.tolist()

print(f"נמצאו {len(columns_to_drop)} עמודות להסרה.")
df_cleaned = df_raw.drop(columns=columns_to_drop)
print("\n סיכום הסרת העמודות ")
print(f"צורת הנתונים המקורית: {df_raw.shape}")
print(f"צורת הנתונים לאחר הסרת עמודות: {df_cleaned.shape}")
print(f"סה\"כ הוסרו {df_raw.shape[1] - df_cleaned.shape[1]} עמודות.")

remaining_missing = df_cleaned.isnull().sum().sort_values(ascending=False)
print(remaining_missing[remaining_missing > 0].head(15))

df_cleaned.to_csv('speed_dating_partially_cleaned.csv', index=False)
print("\n'speed_dating_partially_cleaned.csv'")
