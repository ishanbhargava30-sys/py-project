#Step 1: Import Libraries and Load the Dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the insurance dataset
df = pd.read_csv("insurance.csv")
# Display the first few rows
print(df.head())

#Step 2: Check Shape and Data Types
print("Shape of the dataset:", df.shape)

print("\nData Types & Information:")
print(df.info())

#Step 3: Check and Handle Missing Values
print("Missing values per column:\n", df.isnull().sum())


#Step 4: Explore Relationships (Categorical Count Plots & Numerical Scatter Plots)
# Set visual style
sns.set(style="whitegrid")

# Count plots for categorical columns
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.countplot(x="sex", data=df)
plt.subplot(1, 3, 2)
sns.countplot(x="smoker", data=df)
plt.subplot(1, 3, 3)
sns.countplot(x="region", data=df)
plt.tight_layout()
plt.show()

# Scatter plots for numerical features vs target ('charges')
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.scatterplot(x="age", y="charges", hue="smoker", data=df)
plt.title("Age vs Charges")

plt.subplot(1, 2, 2)
sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df)
plt.title("BMI vs Charges")
plt.tight_layout()
plt.show()

#Step 5: Feature vs Feature Visualization (Correlation & Pairplot)
# Correlation matrix for numerical columns
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

#Step 6: Analyze Premium Charges for Smokers vs. Non-Smokers Over Aging
# Pairplot to see multi-variable relationships
sns.pairplot(df, hue="smoker", palette="Set2")
plt.show()

# Line plot or scatter plot with regression trend lines for Age vs Charges split by Smoker status
plt.figure(figsize=(10, 6))
sns.lmplot(
    x="age", y="charges", hue="smoker", data=df, aspect=1.5, palette="Set1"
)
plt.title("Charges vs Age Grouped by Smoker Status")
plt.show()