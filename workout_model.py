import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

# 1. Load and clean data
df_raw = pd.read_csv('FitNotes_Export_apr6.csv')
df_clean = df_raw.dropna(subset=['Weight', 'Reps']).copy()
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
# Epley Formula for 1RM
df_clean['Estimated_1RM'] = df_clean['Weight'] * (1 + 0.0333 * df_clean['Reps'])

# 2. Group and calculate lag features
df = df_clean.groupby(['Date', 'Exercise', 'Category']).agg(
    Sets=('Reps', 'count'),
    Avg_Reps=('Reps', 'mean'),
    Max_Weight=('Weight', 'max'),
    Session_Max_1RM=('Estimated_1RM', 'max')
).reset_index()

df = df.sort_values(by=['Exercise', 'Date'])
df['Days_Since_Last'] = df.groupby('Exercise')['Date'].diff().dt.days
df['Days_Since_Last'] = df['Days_Since_Last'].fillna(14)
df['Previous_1RM'] = df.groupby('Exercise')['Session_Max_1RM'].shift(1)
df['Previous_1RM'] = df['Previous_1RM'].fillna(df['Session_Max_1RM'])

category_map = df_clean.groupby('Category')['Exercise'].unique().to_dict()
df_encoded = pd.get_dummies(df, columns=['Exercise'])

# 3. Chronological Split
df_encoded = df_encoded.sort_values(by='Date')
inputs = df_encoded.drop(columns=['Date', 'Category', 'Max_Weight', 'Session_Max_1RM'])
outputs = df_encoded[['Max_Weight']]

split_idx = int(len(df_encoded) * 0.8)
inTrain, inTest = inputs.iloc[:split_idx], inputs.iloc[split_idx:]
outTrain, outTest = outputs.iloc[:split_idx], outputs.iloc[split_idx:]

scaler = StandardScaler()
in_trained_scaled = scaler.fit_transform(inTrain)
in_test_scaled = scaler.transform(inTest)

# 4. Neural Network Training
model = Sequential([
    Dense(64, activation='relu', input_shape=(in_trained_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear') 
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Training the model---")
model.fit(in_trained_scaled, outTrain, epochs=100, verbose=0)
print("Training complete!")

# ---------------------------------------------------------
# INTERACTIVE TERMINAL MENU WITH SMART ADJUSTMENTS
# ---------------------------------------------------------
categories = sorted(list(category_map.keys()))

while True:
    print("\n" + "="*30 + "\nWORKOUT PREDICTOR\n" + "="*30)
    for i, cat in enumerate(categories, 1):
        print(f"{i}. {cat}")
    print("0. Exit")
    
    cat_choice = input("\nEnter choice: ")
    if cat_choice == '0': break
        
    try:
        selected_cat = categories[int(cat_choice) - 1]
    except (ValueError, IndexError): continue
        
    while True:
        exercises = sorted(list(category_map[selected_cat]))
        print(f"\n--- {selected_cat.upper()} ---")
        for i, ex in enumerate(exercises, 1):
            print(f"{i}. {ex}")
        print("B. Go Back")
        
        ex_choice = input("\nEnter choice: ")
        if ex_choice.upper() == 'B': break
            
        try:
            selected_ex = exercises[int(ex_choice) - 1]
            target_sets = float(input(f"Target SETS: "))
            target_reps = float(input(f"Target REPS: "))
        except (ValueError, IndexError): continue

        # --- PREDICTION LOGIC ---
        ex_history = df[df['Exercise'] == selected_ex]
        last_1rm = ex_history.iloc[-1]['Session_Max_1RM'] if not ex_history.empty else 100.0
        days_since = (pd.Timestamp.now() - ex_history.iloc[-1]['Date']).days if not ex_history.empty else 14
            
        sim_input = pd.DataFrame(np.zeros((1, len(inputs.columns))), columns=inputs.columns)
        sim_input.loc[0, ['Previous_1RM', 'Days_Since_Last', 'Sets', 'Avg_Reps']] = [last_1rm, days_since, target_sets, target_reps]
        
        if f'Exercise_{selected_ex}' in sim_input.columns:
            sim_input.at[0, f'Exercise_{selected_ex}'] = 1.0
            
        weight_pred = model.predict(scaler.transform(sim_input), verbose=0)[0][0]

        # --- SMART ADJUSTMENT MATH ---
        # 1. Calculate the intensity (1RM) the model wants you to hit
        target_intensity_1rm = weight_pred * (1 + 0.0333 * target_reps)

        # 2. Find nearest 2.5lb increments
        low_w = (weight_pred // 2.5) * 2.5
        high_w = low_w + 2.5

        # 3. Calculate reps needed for each weight to match that intensity
        def calc_reps(t_1rm, w):
            return max(1, round(((t_1rm / w) - 1) / 0.0333))

        reps_low = calc_reps(target_intensity_1rm, low_w)
        reps_high = calc_reps(target_intensity_1rm, high_w)

        print(f"\n>>> RECOMMENDATION FOR {selected_ex.upper()} <<<")
        print(f"Model predicted: {weight_pred:.1f} lbs for {int(target_reps)} reps")
        print(f"\nTo match this intensity with available weights:")
        print(f"Option A: {low_w} lbs for {reps_low} reps")
        print(f"Option B: {high_w} lbs for {reps_high} reps")
        
        input("\nPress Enter to continue...")
        break