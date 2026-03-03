# ===============================================
# SUPERVISED LEARNING - CHIP POWER PREDICTION
# INDUSTRIAL VISUALIZATION VERSION
# ===============================================

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 3. Load Dataset
path = "/content/drive/MyDrive/chip_supervised.csv"
data = pd.read_csv(path)

print("Dataset Shape:", data.shape)
print(data.head())

# 4. Feature Selection
X = data[['Frequency_MHz', 'Voltage_V', 'Area_mm2', 'Temperature_C']]
y = data['Power_Watts']

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
print("\nModel Performance")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# ---------------------------------------------------
# VISUALIZATION 1: Actual vs Predicted
# ---------------------------------------------------
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Power (Watts)")
plt.ylabel("Predicted Power (Watts)")
plt.title("Actual vs Predicted Power\n(Linear Regression Model)")
plt.show()

# ---------------------------------------------------
# VISUALIZATION 2: Frequency vs Power
# ---------------------------------------------------
plt.figure()
plt.scatter(data['Frequency_MHz'], data['Power_Watts'])
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (Watts)")
plt.title("Impact of Frequency on Power")
plt.show()

# ---------------------------------------------------
# VISUALIZATION 3: Area vs Power
# ---------------------------------------------------
plt.figure()
plt.scatter(data['Area_mm2'], data['Power_Watts'])
plt.xlabel("Chip Area (mm²)")
plt.ylabel("Power (Watts)")
plt.title("Impact of Area on Power")
plt.show()

# ---------------------------------------------------
# VISUALIZATION 4: Residual Error Distribution
# ---------------------------------------------------
residuals = y_test - y_pred
plt.figure()
plt.hist(residuals, bins=8)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Residual Error Distribution")
plt.show()
