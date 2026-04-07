import pandas as pd
import numpy as np

# loads the csv data
df = pd.read_csv('FitNotes_Export_apr6.csv')

# Filter out non-resistance exercises
df_clean = df.dropna(subset=['Weight', 'Reps']).copy()
df_clean['Date'] = pd.to_datetime(df_clean['Date'])

# Use Epley Formula for 1 rep max
df_clean['Estimated_1RM'] = df_clean['Weight'] * (1 + 0.0333 * df_clean['Reps'])

# Group by Date and Exercise to summarize each session
workout_summary = df_clean.groupby(['Date', 'Exercise', 'Category']).agg(
    Sets=('Reps', 'count'),
    Avg_Reps=('Reps', 'mean'),
    Max_Weight=('Weight', 'max'),
    Session_Max_1RM=('Estimated_1RM', 'max')
).reset_index()

# Sort by Exercise and Date to calculate historical features
workout_summary = workout_summary.sort_values(by=['Exercise', 'Date'])

# Calculate time between workouts
workout_summary['Days_Since_Last'] = workout_summary.groupby('Exercise')['Date'].diff().dt.days
workout_summary['Days_Since_Last'] = workout_summary['Days_Since_Last'].fillna(14) # Default 14 days for the first session

# Calculate previous 1 rep max
workout_summary['Previous_1RM'] = workout_summary.groupby('Exercise')['Session_Max_1RM'].shift(1)
workout_summary['Previous_1RM'] = workout_summary['Previous_1RM'].fillna(workout_summary['Session_Max_1RM'])

# Round values
workout_summary['Avg_Reps'] = workout_summary['Avg_Reps'].round(1)
workout_summary['Session_Max_1RM'] = workout_summary['Session_Max_1RM'].round(1)
workout_summary['Previous_1RM'] = workout_summary['Previous_1RM'].round(1)

# Save data
workout_summary.to_csv('Processed_Workout_Data.csv', index=False)
print("Data successfully processed and saved to Processed_Workout_Data.csv")
print(workout_summary.head(10))