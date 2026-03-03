# ==============================================
# CHIP POWER PREDICTION USING ML (ASIC STYLE)
# ==============================================

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------
# 1. Generate Synthetic ASIC Manufacturing Data
# ----------------------------------------------

np.random.seed(42)

num_samples = 300

# Frequency in MHz (typical 500 MHz to 3000 MHz)
frequency = np.random.uniform(500, 3000, num_samples)

# Area in mm² (small IP block to large SoC block)
area = np.random.uniform(10, 200, num_samples)

# Realistic power equation (Dynamic power approximation)
# P ≈ C * V^2 * f + Leakage
# For simulation: Power = 0.00005*f*area + small_noise

power = 0.00005 * frequency * area + np.random.normal(0, 2, num_samples)

# Create dataframe
data = pd.DataFrame({
    'Frequency_MHz': frequency,
    'Area_mm2': area,
    'Power_Watts': power
})

print("Sample Data:")
print(data.head())

# ----------------------------------------------
# 2. Train Machine Learning Model
# ----------------------------------------------

X = data[['Frequency_MHz', 'Area_mm2']]
y = data['Power_Watts']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------------
# 3. Model Evaluation
# ----------------------------------------------

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

print("\nModel Coefficients:")
print("Frequency Coefficient:", model.coef_[0])
print("Area Coefficient:", model.coef_[1])
print("Intercept:", model.intercept_)

# ----------------------------------------------
# 4. Visualization 1: Frequency vs Power
# ----------------------------------------------

plt.figure()
plt.scatter(data['Frequency_MHz'], data['Power_Watts'])
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (Watts)")
plt.title("Frequency vs Power")
plt.show()

# ----------------------------------------------
# 5. Visualization 2: Area vs Power
# ----------------------------------------------

plt.figure()
plt.scatter(data['Area_mm2'], data['Power_Watts'])
plt.xlabel("Area (mm²)")
plt.ylabel("Power (Watts)")
plt.title("Area vs Power")
plt.show()

# ----------------------------------------------
# 6. 3D Visualization (Frequency, Area, Power)
# ----------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    data['Frequency_MHz'],
    data['Area_mm2'],
    data['Power_Watts']
)

ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Area (mm²)")
ax.set_zlabel("Power (Watts)")
ax.set_title("3D View: Frequency vs Area vs Power")

plt.show()

# ----------------------------------------------
# 7. Predict New Chip Power
# ----------------------------------------------

new_chip = np.array([[2000, 120]])  # 2000 MHz, 120 mm²
predicted_power = model.predict(new_chip)

print("\nPredicted Power for 2000 MHz & 120 mm²:")
print(predicted_power[0], "Watts")
