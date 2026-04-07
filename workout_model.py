import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore') # get rid of pandas warnings


df_raw = pd.read_csv('FitNotes_Export_apr6.csv')    # read file
df_clean = df_raw.dropna(subset=['Weight', 'Reps']).copy()  # copy weights and reps
df_clean['Date'] = pd.to_datetime(df_clean['Date'])         # extract dates
df_clean['Estimated_1RM'] = df_clean['Weight'] * (1 + 0.0333 * df_clean['Reps'])    # calculate 1rm

# clean and group by date, exercise, and category
df = df_clean.groupby(['Date', 'Exercise', 'Category']).agg(
    Sets=('Reps', 'count'),
    Avg_Reps=('Reps', 'mean'),
    Max_Weight=('Weight', 'max'),
    Session_Max_1RM=('Estimated_1RM', 'max')
).reset_index()

# sort the data
df = df.sort_values(by=['Exercise', 'Date'])

# calculate days since last workout and exercise
df['Days_Since_Last'] = df.groupby('Exercise')['Date'].diff().dt.days
df['Days_Since_Last'] = df['Days_Since_Last'].fillna(14)

# calculate pervious 1rm for the exercise
df['Previous_1RM'] = df.groupby('Exercise')['Session_Max_1RM'].shift(1)
df['Previous_1RM'] = df['Previous_1RM'].fillna(df['Session_Max_1RM'])

# encode per exercise for tracking
df_encoded = pd.get_dummies(df, columns=['Exercise'])


inputs = df_encoded.drop(columns=['Date', 'Category', 'Sets', 'Avg_Reps', 'Max_Weight', 'Session_Max_1RM'])
outputs = df_encoded[['Sets', 'Avg_Reps', 'Max_Weight']]

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

print("\n Predicting Lat PD ")
simulated_input = pd.DataFrame(np.zeros((1, len(inputs.columns))), columns=inputs.columns)

simulated_input.at[0, 'Previous_1RM'] = 110.0 
simulated_input.at[0, 'Days_Since_Last'] = 4.0

if 'Exercise_Lat Pulldown' in simulated_input.columns:
    simulated_input.at[0, 'Exercise_Lat Pulldown'] = 1.0

# Scale and Predict
simulated_input_scaled = scaler.transform(simulated_input)
prediction = model.predict(simulated_input_scaled)

print(f"Recommended Sets: {prediction[0][0]:.1f}")
print(f"Recommended Reps: {prediction[0][1]:.1f}")
print(f"Recommended Weight: {prediction[0][2]:.1f} lbs")