import pandas as pd
import numpy as np

def calculate_hybrid_1rm(weight, reps):
    if reps <= 0: return 0
    if reps <= 6: return weight / (1.0278 - 0.0278 * reps) # Brzycki
    elif reps <= 11: return weight * (1 + 0.0333 * reps) # Epley
    else: return (100 * weight) / (52.2 + 41.9 * np.exp(-0.055 * reps)) # Mayhew

df = pd.read_csv('FitNotes_Export_apr6.csv')
df.columns = df.columns.str.strip()
df_clean = df.dropna(subset=['Weight', 'Reps']).copy()
df_clean['Date'] = pd.to_datetime(df_clean['Date'])

df_clean['Estimated_1RM'] = df_clean.apply(lambda x: calculate_hybrid_1rm(x['Weight'], x['Reps']), axis=1)

workout_summary = df_clean.groupby(['Date', 'Exercise', 'Category']).agg(
    Sets=('Reps', 'count'),
    Avg_Reps=('Reps', 'mean'),
    Max_Weight=('Weight', 'max'),
    Session_Max_1RM=('Estimated_1RM', 'max')
).reset_index()

workout_summary = workout_summary.sort_values(by=['Exercise', 'Date'])
workout_summary['Days_Since_Last'] = workout_summary.groupby('Exercise')['Date'].diff().dt.days.fillna(14)
workout_summary['Previous_1RM'] = workout_summary.groupby('Exercise')['Session_Max_1RM'].shift(1).fillna(workout_summary['Session_Max_1RM'])
# Track last reps to know if we graduated the weight
workout_summary['Last_Avg_Reps'] = workout_summary.groupby('Exercise')['Avg_Reps'].shift(1).fillna(8)

workout_summary.to_csv('Processed_Workout_Data.csv', index=False)
print("Autonomous features processed.")