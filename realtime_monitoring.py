# ==============================================
# Predictive Silicon Intelligence
# Real-Time ASIC Manufacturing Monitoring
# ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output

# For consistent results
np.random.seed(42)

# Create empty dataframe
df = pd.DataFrame(columns=[
    "Time",
    "Temperature_C",
    "Voltage_V",
    "Yield_Percent",
    "Throughput_Units"
])

# Thresholds for anomaly detection
TEMP_THRESHOLD = 82
VOLTAGE_LOW_THRESHOLD = 0.92
YIELD_LOW_THRESHOLD = 92

print("Starting Real-Time ASIC Manufacturing Monitoring...\n")

# Simulate real-time data stream
for t in range(30):

    # Generate realistic manufacturing data
    temperature = np.random.normal(75, 3)
    voltage = np.random.normal(1.0, 0.03)
    yield_percent = np.random.normal(95, 1)
    throughput = np.random.normal(500, 20)

    # Append to dataframe
    new_row = pd.DataFrame([[t, temperature, voltage, yield_percent, throughput]],
                           columns=df.columns)

    df = pd.concat([df, new_row], ignore_index=True)

    # Clear previous output for live effect
    clear_output(wait=True)

    # Plotting
    plt.figure(figsize=(14,10))

    # Temperature Plot
    plt.subplot(2,2,1)
    plt.plot(df["Time"], df["Temperature_C"], marker='o')
    plt.axhline(TEMP_THRESHOLD, linestyle='--')
    plt.title("Temperature Trend (°C)")
    plt.xlabel("Time")
    plt.ylabel("Temperature")

    # Voltage Plot
    plt.subplot(2,2,2)
    plt.plot(df["Time"], df["Voltage_V"], marker='o')
    plt.axhline(VOLTAGE_LOW_THRESHOLD, linestyle='--')
    plt.title("Voltage Trend (V)")
    plt.xlabel("Time")
    plt.ylabel("Voltage")

    # Yield Plot
    plt.subplot(2,2,3)
    plt.plot(df["Time"], df["Yield_Percent"], marker='o')
    plt.axhline(YIELD_LOW_THRESHOLD, linestyle='--')
    plt.title("Yield Percentage Trend")
    plt.xlabel("Time")
    plt.ylabel("Yield %")

    # Throughput Plot
    plt.subplot(2,2,4)
    plt.plot(df["Time"], df["Throughput_Units"], marker='o')
    plt.title("Throughput Trend (Units)")
    plt.xlabel("Time")
    plt.ylabel("Units")

    plt.tight_layout()
    plt.show()

    time.sleep(0.5)

# Save CSV file
df.to_csv("asic_real_time_monitoring.csv", index=False)

print("\nSimulation Completed.")
print("CSV File Saved as: asic_real_time_monitoring.csv")

# ---------------------------------------
# Anomaly Detection After Simulation
# ---------------------------------------

anomalies = df[
    (df["Temperature_C"] > TEMP_THRESHOLD) |
    (df["Voltage_V"] < VOLTAGE_LOW_THRESHOLD) |
    (df["Yield_Percent"] < YIELD_LOW_THRESHOLD)
]

print("\nDetected Anomalies:")
print(anomalies)

print("\nTotal Anomalies Detected:", len(anomalies))
