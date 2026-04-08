import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# --- 1. Helper Functions ---
def calculate_hybrid_1rm(weight, reps):
    if reps <= 0: return 0
    if reps <= 6: return weight / (1.0278 - 0.0278 * reps)
    elif reps <= 11: return weight * (1 + 0.0333 * reps)
    else: return (100 * weight) / (52.2 + 41.9 * np.exp(-0.055 * reps))

def get_weight(one_rm, reps): 
    return one_rm / (1 + 0.0333 * reps)

# --- 2. Data Processing & Model Pipeline ---
@st.cache_data
def load_and_train_model(file):
    # Process Raw FitNotes CSV
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df_clean = df.dropna(subset=['Weight', 'Reps']).copy()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean['Estimated_1RM'] = df_clean.apply(lambda x: calculate_hybrid_1rm(x['Weight'], x['Reps']), axis=1)
    
    # This filters out warmups so Streamlit only averages your heaviest sets
    idx = df_clean.groupby(['Date', 'Exercise'])['Weight'].transform('max') == df_clean['Weight']
    working_sets = df_clean[idx]
    
    # Change df_clean to working_sets here!
    workout_summary = working_sets.groupby(['Date', 'Exercise', 'Category']).agg(
        Sets=('Reps', 'count'),
        Avg_Reps=('Reps', 'mean'),
        Max_Weight=('Weight', 'max'),
        Session_Max_1RM=('Estimated_1RM', 'max')
    ).reset_index()
    
    workout_summary = workout_summary.sort_values(by=['Exercise', 'Date'])
    workout_summary['Days_Since_Last'] = workout_summary.groupby('Exercise')['Date'].diff().dt.days.fillna(14)
    workout_summary['Previous_1RM'] = workout_summary.groupby('Exercise')['Session_Max_1RM'].shift().fillna(workout_summary['Session_Max_1RM'])
    workout_summary['Last_Avg_Reps'] = workout_summary.groupby('Exercise')['Avg_Reps'].shift().fillna(workout_summary['Avg_Reps'])
    
    # Train Model
    df_encoded = pd.get_dummies(workout_summary, columns=['Exercise', 'Category'])
    features_to_drop = ['Date', 'Max_Weight', 'Session_Max_1RM', 'Last_Avg_Reps', 'Avg_Reps', 'Sets']
    X = df_encoded.drop(columns=features_to_drop, errors='ignore')
    y = df_encoded['Session_Max_1RM']
    
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    
    return model, workout_summary, list(X.columns)

st.title("Autonomous AI Workout Model")

# File Uploader
uploaded_file = st.file_uploader("Upload your raw FitNotes CSV Export", type="csv")

if uploaded_file is not None:
    model, workout_summary, feature_cols = load_and_train_model(uploaded_file)
    st.success("Data processed and model trained successfully!")
    
    # UI Layout
    col1, col2 = st.columns(2)
    
    with col1:
        categories = sorted(workout_summary['Category'].unique())
        selected_cat = st.selectbox("Select Category:", categories)
        
    with col2:
        exercises = sorted(workout_summary[workout_summary['Category'] == selected_cat]['Exercise'].unique())
        selected_ex = st.selectbox("Select Exercise:", exercises)
        
    if st.button("Get Recommendation", type="primary"):
        # Prediction Logic
        ex_data = workout_summary[workout_summary['Exercise'] == selected_ex]
        
        if not ex_data.empty:
            last_session = ex_data.iloc[-1]
            last_1rm = last_session['Session_Max_1RM']
            last_days = last_session['Days_Since_Last']
            last_w = last_session['Max_Weight']
            last_reps = last_session['Avg_Reps']
            
            # Setup simulation input array
            sim_input = pd.DataFrame(columns=feature_cols)
            sim_input.loc[0] = 0.0 # Initialize with zeros
            
            sim_input.at[0, 'Days_Since_Last'] = last_days
            sim_input.at[0, 'Previous_1RM'] = last_1rm
            
            if f'Exercise_{selected_ex}' in feature_cols: 
                sim_input.at[0, f'Exercise_{selected_ex}'] = 1.0
            if f'Category_{selected_cat}' in feature_cols: 
                sim_input.at[0, f'Category_{selected_cat}'] = 1.0

            sim_input = sim_input.astype(float)
            pred_1rm = model.predict(sim_input)[0]
            
            # Decision Logic (The 6-10 Sweet Spot)
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

            if target_w == 0: 
                target_w = round(get_weight(pred_1rm, 8) / 2.5) * 2.5
                target_r = 8
                status = "NEW EXERCISE: Baseline"
                
            # Output Display
            st.divider()
            st.subheader(f"SET GOAL: 3 Sets x {target_r} Reps")
            st.metric(label="Target Weight", value=f"{target_w} lbs")
            st.info(f"STATUS: {status}")
        else:
            st.warning("No data found for this exercise.")
else:
    st.info("Awaiting CSV upload...")