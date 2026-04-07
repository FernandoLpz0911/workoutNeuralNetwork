import pandas as pd
import numpy as np
import xgboost as xgb

# 1. Load and Train
df = pd.read_csv('Processed_Workout_Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df_encoded = pd.get_dummies(df, columns=['Exercise', 'Category'])

# Features: History only (No target reps/sets to avoid bias)
features_to_drop = ['Date', 'Max_Weight', 'Session_Max_1RM', 'Last_Avg_Reps', 'Avg_Reps', 'Sets']
X = df_encoded.drop(columns=features_to_drop, errors='ignore')
y = df_encoded['Session_Max_1RM']

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X, y)

def get_weight(one_rm, reps): return one_rm / (1 + 0.0333 * reps)

# ---------------------------------------------------------
# SEQUENTIAL AUTONOMOUS ENGINE
# ---------------------------------------------------------
category_map = df.groupby('Category')['Exercise'].unique().to_dict()
categories = sorted(list(category_map.keys()))

while True:
    print("\n" + "\nAutonomous AI Workout Model\n")
    for i, cat in enumerate(categories, 1): print(f"{i}. {cat}")
    print("0. Exit")
    
    cat_choice = input("\nSelect Category: ")
    if cat_choice == '0' or not cat_choice: break
    
    try:
        selected_cat = categories[int(cat_choice)-1]
        
        while True: # Exercise-by-Exercise Loop
            exercises = sorted(list(category_map[selected_cat]))
            print(f"\n--- {selected_cat.upper()} EXERCISES ---")
            for i, ex in enumerate(exercises, 1): print(f"{i}. {ex}")
            print("B. Go Back to Categories")
            
            ex_choice = input("\nSelect Exercise to perform: ")
            if ex_choice.upper() == 'B': break
            
            selected_ex = exercises[int(ex_choice)-1]
            
            # 1. Prediction State
            ex_history = df[df['Exercise'] == selected_ex]
            last_w = ex_history.iloc[-1]['Max_Weight'] if not ex_history.empty else 0
            last_1rm = ex_history.iloc[-1]['Session_Max_1RM'] if not ex_history.empty else 100
            last_reps = ex_history.iloc[-1]['Avg_Reps'] if not ex_history.empty else 8
            
            sim_input = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
            sim_input.loc[0, ['Previous_1RM', 'Days_Since_Last']] = [last_1rm, 7]
            
            if f'Exercise_{selected_ex}' in sim_input.columns: sim_input.at[0, f'Exercise_{selected_ex}'] = 1
            if f'Category_{selected_cat}' in sim_input.columns: sim_input.at[0, f'Category_{selected_cat}'] = 1

            pred_1rm = model.predict(sim_input)[0]
            
            # 2. Decision Logic (The 6-10 Sweet Spot)
            # Threshold Progression: If you hit 10 reps, you graduate.
            if last_reps >= 10:
                target_w = last_w + 2.5
                target_r = 8
                status = "PROGRESSION: Weight Increased"
            elif last_reps < 6:
                target_w = last_w
                target_r = 8
                status = "STABILIZATION: Form Focus"
            else:
                target_w = last_w
                target_r = 10
                status = "VOLUME: Pushing for Graduation"

            # Initial state for brand new exercises
            if target_w == 0: 
                target_w = round(get_weight(pred_1rm, 8) / 2.5) * 2.5
                target_r = 8
                status = "NEW EXERCISE: Baseline"

            # 3. Output Recommendation
            print(f"\n" + "*"*30)
            print(f"Selected: {selected_ex.upper()}")
            print(f"*"*30)
            print(f"SET GOAL: 3 Sets x {target_r} Reps")
            print(f"WEIGHT  : {target_w} lbs")
            print(f"STATUS  : {status}")
            print(f"CAPACITY: {pred_1rm:.1f}lb 1RM Est.")
            print("*"*30)
            
            input("\nPress Enter when set is complete to return...")
            
    except (ValueError, IndexError):
        continue