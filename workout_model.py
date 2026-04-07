import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('Processed_Workout_Data.csv')
df = pd.get_dummies(df, columns=['Category'])


inputs = df.drop(columns=['Date', 'Exercise', 'Sets', 'Avg_Reps', 'Max_Weight', 'Session_Max_1RM'])

outputs = df[['Sets', 'Avg_Reps', 'Max_Weight']]

inTrain, inTest, outTrain, outTest = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# scale the input features so no single metric overpowers the others
scaler = StandardScaler()
in_trained_scaled = scaler.fit_transform(inTrain)
in_test_scaled = scaler.transform(inTest)

# build neural network from 64 nodes in, then 32 node 2nd layer
# output to 3 node for sets, reps, and weight
# uses linear activation for regression
model = Sequential([
    Dense(64, activation='relu', input_shape=(in_trained_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(3, activation='linear')
])

# compile the model
# 'mse' (Mean Squared Error) is best for predicting numbers
# 'adam' is the most reliable optimizer to adjust the weights
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# model training
print("Training the model---")
history = model.fit(in_trained_scaled, outTrain, epochs=100, verbose=0)
print("Training complete!")
loss, mae = model.evaluate(in_test_scaled, outTest, verbose=0)
print(f"\nModel Accuracy Check:")
print(f"On average, the model's predictions are off by: {mae:.2f} units (reps/sets/lbs)")

print("\n--- PREDICTING NEXT BACK WORKOUT ---")
simulated_input = pd.DataFrame(np.zeros((1, len(inputs.columns))), columns=inputs.columns)

simulated_input.at[0, 'Previous_1RM'] = 135.0 
simulated_input.at[0, 'Days_Since_Last'] = 5.0
if 'Category_Back' in simulated_input.columns:
    simulated_input.at[0, 'Category_Back'] = 1.0   # 1 means "Yes, this is a Back exercise"

# Scale and Predict
simulated_input_scaled = scaler.transform(simulated_input)
prediction = model.predict(simulated_input_scaled)

print(f"Recommended Sets: {prediction[0][0]:.1f}")
print(f"Recommended Reps: {prediction[0][1]:.1f}")
print(f"Recommended Weight: {prediction[0][2]:.1f} lbs")