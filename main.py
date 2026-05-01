import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_data, clean_data, split_features_target
from src.model import train_linear_regression, train_random_forest
from src.utils import evaluate_model

# -----------------------------
# DARK THEME SETTINGS
# -----------------------------
plt.style.use("dark_background")
sns.set_theme(style="darkgrid")

plt.rcParams.update({
    "figure.facecolor": "#111111",
    "axes.facecolor": "#111111",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "figure.figsize": (8, 5)
})

# -----------------------------
# STEP 1: CREATE REALISTIC DATASET
# -----------------------------
print("Creating realistic dataset...")

np.random.seed(42)
data_size = 300

area = np.random.randint(500, 3000, data_size)
bedrooms = np.random.randint(1, 5, data_size)
bathrooms = np.random.randint(1, 4, data_size)
parking = np.random.randint(0, 3, data_size)
age = np.random.randint(0, 20, data_size)

price = (
    area * 3000 +
    bedrooms * 500000 +
    bathrooms * 300000 +
    parking * 200000 -
    age * 100000 +
    np.random.randint(-200000, 200000, data_size)
)

data = pd.DataFrame({
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'parking': parking,
    'age': age,
    'price': price
})

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

data.to_csv("data/housing_data.csv", index=False)

# -----------------------------
# STEP 2: LOAD DATA
# -----------------------------
df = load_data("data/housing_data.csv")
df = clean_data(df)

print("\nDataset Preview:")
print(df.head())

# -----------------------------
# STEP 3: EDA
# -----------------------------
print("\nGenerating graphs...")

# 👉 Pairplot with WHITE background (override dark theme)
sns.set_theme(style="white")
pair = sns.pairplot(df, corner=True)
pair.fig.set_facecolor("white")
pair.savefig("images/pairplot.png", dpi=300)
plt.close()

# Reset back to dark theme
sns.set_theme(style="darkgrid")

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='viridis', linewidths=0.5)
plt.title("Correlation Between Features", fontsize=14)
plt.tight_layout()
plt.savefig("images/heatmap.png", dpi=300)
plt.close()

# Price Distribution
plt.figure()
sns.histplot(df['price'], bins=30, kde=True, color="#00ADB5")
plt.title("Distribution of House Prices", fontsize=14)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("images/price_distribution.png", dpi=300)
plt.close()

# -----------------------------
# STEP 4: 3D VISUALIZATION
# -----------------------------
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    df['area'],
    df['bedrooms'],
    df['price'],
    c=df['price'],
    cmap='plasma'
)

ax.set_xlabel('Area')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')

plt.title("3D View: Area vs Bedrooms vs Price")
plt.tight_layout()
plt.savefig("images/3d_plot.png", dpi=300)
plt.close()

# -----------------------------
# STEP 5: SPLIT DATA
# -----------------------------
X, y = split_features_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 6: TRAIN MODELS
# -----------------------------
print("\nTraining models...")

lr_model = train_linear_regression(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

# Save model
joblib.dump(rf_model, "models/model.pkl")

# -----------------------------
# STEP 7: PREDICTIONS
# -----------------------------
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": rf_pred
})

results.to_csv("outputs/predictions.csv", index=False)

# -----------------------------
# STEP 8: EVALUATION
# -----------------------------
evaluate_model(y_test, lr_pred, "Linear Regression")
evaluate_model(y_test, rf_pred, "Random Forest")

# -----------------------------
# STEP 9: PREDICTION GRAPH
# -----------------------------
plt.figure(figsize=(7,7))

sns.scatterplot(x=y_test, y=rf_pred, color="#00ADB5", alpha=0.7)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linestyle='--',
    linewidth=2
)

plt.title("Actual vs Predicted House Prices", fontsize=14)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.tight_layout()
plt.savefig("images/prediction.png", dpi=300)
plt.close()

# -----------------------------
# STEP 10: SAMPLE PREDICTION
# -----------------------------
new_house = np.array([[1500, 3, 2, 1, 5]])

pred_price = rf_model.predict(new_house)

print("\nSample Prediction:")
print("Features:", new_house)
print("Predicted Price:", int(pred_price[0]))