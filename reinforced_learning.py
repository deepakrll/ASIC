# ===============================================
# REINFORCEMENT LEARNING - Q LEARNING
# INDUSTRIAL VISUALIZATION VERSION
# ===============================================

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
path = "/content/drive/MyDrive/chip_reinforcement.csv"
data = pd.read_csv(path)

print("Dataset Shape:", data.shape)
print(data.head())

# Create States
data['Area_State'] = pd.cut(data['Area_mm2'], bins=3, labels=False)

states = 3
actions = 3

Q = np.zeros((states, actions))

learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1

episode_rewards = []

# Training
for episode in range(500):
    total_reward = 0
    
    for index, row in data.iterrows():
        state = int(row['Area_State'])
        
        if np.random.rand() < epsilon:
            action = np.random.randint(actions)
        else:
            action = np.argmax(Q[state])
        
        reward = row['Performance_Index'] - row['Power_Watts']
        total_reward += reward
        
        Q[state, action] += learning_rate * (
            reward + discount_factor * np.max(Q[state]) - Q[state, action]
        )
    
    episode_rewards.append(total_reward)

print("\nFinal Q Table:")
print(Q)

# ---------------------------------------------------
# VISUALIZATION 1: Q Table Heatmap
# ---------------------------------------------------
plt.figure()
plt.imshow(Q)
plt.colorbar()
plt.xlabel("Frequency Level (Action)")
plt.ylabel("Area Category (State)")
plt.title("Learned Q-Table")
plt.show()

# ---------------------------------------------------
# VISUALIZATION 2: Reward Convergence
# ---------------------------------------------------
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Convergence Over Episodes")
plt.show()

# ---------------------------------------------------
# VISUALIZATION 3: Best Action per State
# ---------------------------------------------------
best_actions = np.argmax(Q, axis=1)

plt.figure()
plt.bar(range(states), best_actions)
plt.xlabel("Area Category")
plt.ylabel("Best Frequency Level")
plt.title("Optimal Frequency Level per Area Category")
plt.show()
