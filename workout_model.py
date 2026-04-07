import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# mock data (similar to formatted dataset)
# data: [Age, 1RM_Weight, Sleep_Hours, Is_Hypertrophy, Is_Strength]
X_data = np.array([
    [25, 225, 8.0, 1, 0],  # 25yo, 225 1RM, 8h sleep, Hypertrophy goal
    [30, 315, 6.5, 0, 1],  # 30yo, 315 1RM, 6.5h sleep, Strength goal
    [22, 135, 7.0, 1, 0],  # 22yo, 135 1RM, 7h sleep, Hypertrophy goal
    [40, 405, 8.5, 0, 1],  # 40yo, 405 1RM, 8.5h sleep, Strength goal
    [28, 275, 5.0, 1, 0]   # 28yo, 275 1RM, 5h sleep, Hypertrophy goal
])

# targets: [Optimal_Sets, Optimal_Reps, Optimal_Weight]
y_data = np.array([
    [4, 10, 160],  # 4 sets, 10 reps, 160 lbs
    [5,  3, 275],  # 5 sets,  3 reps, 275 lbs
    [3, 12,  95],  # 3 sets, 12 reps, 95 lbs
    [5,  5, 345],  # 5 sets,  5 reps, 345 lbs
    [3,  8, 185]   # 3 sets,  8 reps, 185 lbs (poor sleep = lighter weight/vol)
])

# format and scale data
# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# scale the input features so no single metric overpowers the others
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# build neural network from 64 nodes in, then 32 node 2nd layer
# output to 3 node for sets, reps, and weight
# uses linear activation for regression
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(3, activation='linear')
])

# compile the model
# 'mse' (Mean Squared Error) is best for predicting numbers
# 'adam' is the most reliable optimizer to adjust the weights
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# model training
print("Training the model---")
history = model.fit(X_train_scaled, y_train, epochs=100, verbose=0)
print("Training complete!")

# test prediction
# prediction: 27yo, 250lb 1RM, 7.5h sleep, Hypertrophy goal
new_user = np.array([[27, 250, 7.5, 1, 0]])
new_user_scaled = scaler.transform(new_user)  # scales inputs

prediction = model.predict(new_user_scaled)
print(f"\nPredicted Workout -> Sets: {prediction[0][0]:.1f}, Reps: {prediction[0][1]:.1f}, Weight: {prediction[0][2]:.1f} lbs")