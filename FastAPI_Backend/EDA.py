import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize

df = pd.read_csv("../Data/dataset.csv", compression='gzip', low_memory=False)

nutri_cols = [
    'Calories','ProteinContent','CarbohydrateContent',
    'FatContent','SodiumContent','FiberContent','SugarContent'
]

df[nutri_cols] = df[nutri_cols].apply(pd.to_numeric, errors='coerce')

df = df.dropna(subset=nutri_cols).reset_index(drop=True)

# Reduce outliers
df_clean = df.copy()
for col in nutri_cols:
    df_clean[col] = winsorize(df[col], limits=[0.05, 0.05])  # trim 5%

# Calories Distribution
plt.figure(figsize=(9,5))
sns.histplot(df_clean['Calories'], bins=40, kde=True, color="#d62828", alpha=0.7)
plt.title("Calories Distribution", fontsize=16, weight="bold")
plt.xlabel("Calories")
plt.ylabel("Density")
plt.show()

# Boxplot of Key Nutrition Features
plt.figure(figsize=(10,5))
sns.boxplot(data=df_clean[nutri_cols], palette="Set2")
plt.title("Distribution of Nutrition Attributes", fontsize=16, weight="bold")
plt.xticks(rotation=20)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df_clean[nutri_cols].corr(), annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.6)
plt.title("Correlation Between Nutrition Features", fontsize=16, weight="bold")
plt.show()