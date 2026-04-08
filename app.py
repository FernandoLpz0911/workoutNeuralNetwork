import streamlit as st
import pandas as pd
import joblib
from pipeline import run_pipeline

# --- 1. Helper Functions ---
def get_weight(one_rm, reps): 
    return one_rm / (1 + 0.0333 * reps)

# --- 2. Load Pre-trained Assets ---
@st.cache_resource
def load_assets():
    model = joblib.load('xgb_model.joblib')
    feature_cols = joblib.load('feature_cols.joblib')
    workout_summary = pd.read_csv('Processed_Workout_Data.csv')
    return model, feature_cols, workout_summary

st.set_page_config(page_title="AI Workout Model", layout="wide")
st.title("Autonomous AI Workout Model")

# Try to load assets
try:
    model, feature_cols, workout_summary = load_assets()
    assets_loaded = True
except FileNotFoundError:
    assets_loaded = False

# --- 3. Sidebar for Data Upload & Retraining ---
with st.sidebar:
    st.header("Update Data")
    uploaded_file = st.file_uploader("Upload FitNotes CSV", type="csv")
    
    if uploaded_file is not None:
        if st.button("Train Model with New Data", type="primary", use_container_width=True):
            with st.spinner("Processing FitNotes data and training XGBoost..."):
                # Pass the uploaded file to the pipeline!
                run_pipeline(uploaded_file)
                st.cache_resource.clear() 
            st.success("Training complete!")
            st.rerun()

# --- 4. User Interface ---
if not assets_loaded:
    st.warning("Model and data assets not found. Please upload your FitNotes CSV in the sidebar to initialize the AI.")
else:
    st.success("Model loaded and ready!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        categories = sorted(workout_summary['Category'].unique())
        selected_cat = st.selectbox("Select Category:", categories)
        
    with col2:
        exercises = sorted(workout_summary[workout_summary['Category'] == selected_cat]['Exercise'].unique())
        selected_ex = st.selectbox("Select Exercise:", exercises)
        
    if st.button("Get Recommendation", type="primary"):
        ex_data = workout_summary[workout_summary['Exercise'] == selected_ex]
        
        if not ex_data.empty:
            last_session = ex_data.iloc[-1]
            last_1rm = last_session['Session_Max_1RM']
            last_days = last_session['Days_Since_Last']
            last_w = last_session['Max_Weight']
            last_reps = last_session['Avg_Reps']
            last_volume = last_session['Volume_Load'] 
            
            # Setup simulation input array
            sim_input = pd.DataFrame(columns=feature_cols)
            sim_input.loc[0] = 0.0 
            
            sim_input.at[0, 'Days_Since_Last'] = last_days
            sim_input.at[0, 'Previous_1RM'] = last_1rm
            sim_input.at[0, 'Prev_Volume_Load'] = last_volume 
            
            if f'Exercise_{selected_ex}' in feature_cols: 
                sim_input.at[0, f'Exercise_{selected_ex}'] = 1.0
            if f'Category_{selected_cat}' in feature_cols: 
                sim_input.at[0, f'Category_{selected_cat}'] = 1.0

            sim_input = sim_input.astype(float)
            pred_1rm = model.predict(sim_input)[0]
            
            # Decision Logic 
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
                
            st.divider()
            st.subheader(f"SET GOAL: 3 Sets x {target_r} Reps")
            st.metric(label="Target Weight", value=f"{target_w} lbs")
            st.info(f"STATUS: {status}")
        else:
            st.warning("No data found for this exercise.")