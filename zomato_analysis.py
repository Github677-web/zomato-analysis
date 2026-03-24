# ================= ZOMATO DATA ANALYSIS PROJECT =================

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Load dataset
df = pd.read_csv("zomato.csv")

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

print("First 5 rows:\n", df.head())


# Step 3: Explore data
print("\nShape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nInfo:\n")
print(df.info())


# Step 4: Data Cleaning

# Drop unnecessary columns safely
df.drop(['url', 'phone'], axis=1, inplace=True, errors='ignore')

# Clean 'rate' column
df['rate'] = df['rate'].astype(str).str.replace('/5', '', regex=False)
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

# Clean cost column
cost_col = 'approx_cost(for two people)'
df[cost_col] = df[cost_col].astype(str).str.replace(',', '', regex=False)
df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')

# Drop rows with important missing values only
df.dropna(subset=['rate', cost_col], inplace=True)


# Step 5: Basic Analysis
print("\nTop 10 Locations:\n", df['location'].value_counts().head(10))

print("\nAverage Rating (Online Order):\n",
      df.groupby('online_order')['rate'].mean())


# Step 6: Visualization
sns.set(style="whitegrid")

# 1. Rating distribution (existing)
plt.figure(figsize=(8, 5))
sns.histplot(df['rate'], bins=20)
plt.title("Rating Distribution")
plt.show()

# 2. Top locations (existing)
plt.figure(figsize=(8, 5))
df['location'].value_counts().head(10).sort_values().plot(kind='barh')
plt.title("Top Locations")
plt.show()

# 3. Cost vs Rating (NEW)
plt.figure(figsize=(8,5))
sns.scatterplot(x='approx_cost(for two people)', y='rate', data=df)
plt.title("Cost vs Rating")
plt.show()

# 4. Online Order Pie Chart (NEW)
plt.figure(figsize=(6,6))
df['online_order'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Online Order Availability")
plt.ylabel("")
plt.show()

# 5. Top Cuisines (NEW)
plt.figure(figsize=(8,5))
df['cuisines'].value_counts().head(10).plot(kind='bar')
plt.title("Top Cuisines")
plt.xticks(rotation=45)
plt.show()


# Step 7: Advanced Analysis

# Top restaurants
top_restaurants = df.sort_values(by=['rate', 'votes'], ascending=False)
print("\nTop 10 Restaurants:\n",
      top_restaurants[['name', 'rate', 'votes']].head(10))

# Popular cuisines
print("\nTop 10 Cuisines:\n",
      df['cuisines'].value_counts().head(10))

# Best locations by rating
location_rating = df.groupby('location')['rate'].mean().sort_values(ascending=False)
print("\nBest Locations by Rating:\n", location_rating.head(10))

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Online order vs rating
plt.figure(figsize=(6, 4))
sns.boxplot(x='online_order', y='rate', data=df)
plt.title("Online Order vs Rating")
plt.show()


# Step 8: Save cleaned dataset
df.to_csv("cleaned_zomato.csv", index=False)

# ---------------- INSIGHTS ----------------
print("\nINSIGHTS:")
print("1. Most restaurants are concentrated in popular locations.")
print("2. Online orderinggit status has some influence on ratings.")
print("3. Expensive restaurants are not always highly rated.")
print("4. Certain cuisines dominate the market.")
