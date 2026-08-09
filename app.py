import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64

# ==========================================================
# PAGE INITIALIZATION & THEME SETTING
# ==========================================================
st.set_page_config(page_title="Flight Delay Expert", page_icon="✈️", layout="wide")

# ==========================================================
# HELPER FUNCTIONS FOR LOCAL IMAGES
# ==========================================================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_jpeg_as_page_bg(jpeg_file):
    bin_str = get_base64_of_bin_file(jpeg_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: local;
    }}
    
    [data-testid="stAppViewContainer"] {{
        background-color: rgba(15, 23, 42, 0.85);
        border-radius: 10px;
        padding: 20px;
    }}

    h1, h2, h3 {{
        color: #38bdf8 !important;
    }}
    .stMarkdown, p, label {{
        color: #f8fafc !important;
        font-weight: 500;
    }}

    div[data-baseweb="select"] > div {{
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white;
    }}
    input {{
        color: white !important;
    }}
    
    /* Strict Circular Styling for the Center Icon */
    .centered-icon-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 32px;
    }}
    .circular-icon {{
        width: 55px;
        height: 55px;
        border-radius: 50%;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
        border: 2px solid #38bdf8;
        box-shadow: 0 4px 10px rgba(56, 189, 248, 0.4);
        background-color: #0f172a;
    }}
    .circular-icon img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 50%;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ==========================================================
# CONFIGURATION & BACKGROUND IMAGE LOADING WITH EXCEPTION HANDLING
# ==========================================================
bg_image_file = 'background.jpeg'

try:
    set_jpeg_as_page_bg(bg_image_file)
except FileNotFoundError:
    st.error(f"Background image file '{bg_image_file}' not found. Please place the JPEG in your project folder.")
    st.info("Proceeding with default theme until background is fixed.")
except Exception as e:
    st.error(f"An unexpected error occurred while loading the background image: {e}")
    st.info("Proceeding with default theme.")

# ==========================================================
# 1. LOAD MODEL BRAIN ARTIFACTS & ACTIVE SCHEDULE MAPS
# ==========================================================
@st.cache_resource
def load_all_assets():
    # FIX: Updated to match your exact saved model filename ('flight_delay_model.pkl')
    model_assets = joblib.load('flight_delay_model.pkl')
    schedule_lookup = joblib.load('flight_schedule_lookup.pkl')
    airport_df = pd.read_csv('airports.csv')
    
    airport_df['DISPLAY_NAME'] = airport_df['CITY'] + " - " + airport_df['AIRPORT'] + " (" + airport_df['IATA_CODE'] + ")"
    
    code_to_display = pd.Series(airport_df.DISPLAY_NAME.values, index=airport_df.IATA_CODE).to_dict()
    display_to_code = pd.Series(airport_df.IATA_CODE.values, index=airport_df.DISPLAY_NAME).to_dict()
    
    return (
        model_assets['model'], 
        model_assets['expected_columns'], 
        schedule_lookup, 
        code_to_display, 
        display_to_code
    )

model, expected_columns, schedule_df, code_to_display, display_to_code = load_all_assets()

AIRLINE_NAMES = {
    "AS": "Alaska Airlines", "B6": "JetBlue Airways", "DL": "Delta Air Lines", 
    "EV": "ExpressJet Airlines", "F9": "Frontier Airlines", "HA": "Hawaiian Airlines", 
    "MQ": "Envoy Air", "NK": "Spirit Airlines", "OO": "SkyWest Airlines", 
    "UA": "United Airlines", "US": "US Airways", "VX": "Virgin America", "WN": "Southwest Airlines"
}

# Title Header
st.title("✈️ Smart Flight Delay Predictor")
st.markdown(
    "Welcome to your professional predictive dashboard. Select your route, date of travel, "
    "and available flight time to generate a live arrival delay forecast."
)
st.markdown("---")

# ==========================================================
# 2. STEP 1: ROUTE & CARRIER SELECTION
# ==========================================================
st.subheader("🛫 Route & Carrier Selection")

col1, col_icon, col2 = st.columns([5, 1, 5])

with col1:
    available_origins = sorted([code_to_display[c] for c in schedule_df['origin_airport'].unique() if c in code_to_display])
    selected_origin_display = st.selectbox("Departure Airport", options=available_origins)
    origin_code = display_to_code[selected_origin_display]

with col_icon:
    try:
        icon_base64 = get_base64_of_bin_file('icon.jpeg')
        st.markdown(f'''
            <div class="centered-icon-container">
                <div class="circular-icon">
                    <img src="data:image/jpeg;base64,{icon_base64}" alt="Flight Icon">
                </div>
            </div>
        ''', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("icon.jpeg missing")

with col2:
    filtered_dests = schedule_df[schedule_df['origin_airport'] == origin_code]['destination_airport'].unique()
    available_dests = sorted([code_to_display[c] for c in filtered_dests if c in code_to_display])
    selected_dest_display = st.selectbox("Arrival Airport", options=available_dests)
    dest_code = display_to_code[selected_dest_display]

st.markdown("<br>", unsafe_allow_html=True)

# Carrier selection right underneath route
filtered_airlines = schedule_df[(schedule_df['origin_airport'] == origin_code) & (schedule_df['destination_airport'] == dest_code)]['airline'].unique()
available_airlines = sorted([AIRLINE_NAMES[a] for a in filtered_airlines if a in AIRLINE_NAMES])
selected_airline_str = st.selectbox("Available Airlines on this Route", options=available_airlines)

inverse_airline_lookup = {v: k for k, v in AIRLINE_NAMES.items()}
airline_code = inverse_airline_lookup[selected_airline_str]

# ==========================================================
# 3. STEP 2: DATE OF TRAVEL & DYNAMIC TIME SLOTS SIDE BY SIDE
# ==========================================================
st.markdown("---")
st.subheader("📅 Schedule & Timing Selection")

col3, col4 = st.columns(2)

with col3:
    travel_date = st.date_input("Select your departure date", value=pd.to_datetime("2026-06-08"))

with col4:
    raw_timings = schedule_df[
        (schedule_df['origin_airport'] == origin_code) & 
        (schedule_df['destination_airport'] == dest_code) & 
        (schedule_df['airline'] == airline_code)
    ]['scheduled_departure'].values[0]

    def format_military_time(val):
        val = int(val)
        hours = val // 100
        minutes = val % 100
        return f"{hours:02d}:{minutes:02d}"

    timing_options = sorted(list(raw_timings))
    timing_display_map = {format_military_time(t): t for t in timing_options}

    selected_time_str = st.selectbox(
        f"Available Departure Times for {travel_date.strftime('%B %d, %Y')}", 
        options=list(timing_display_map.keys())
    )
    scheduled_departure = timing_display_map[selected_time_str]

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# 4. PREDICTION ENGINE & FEATURE ENGINEERING TRANSLATION
# ==========================================================
if st.button("Predict Flight Arrival Status", type="primary"):
    month = travel_date.month
    day_of_week = travel_date.weekday() + 1
    
    input_data = {col: [0.0] for col in expected_columns}
    
    if 'SCHEDULED_DEPARTURE' in input_data:
        input_data['SCHEDULED_DEPARTURE'] = [float(scheduled_departure)]
        
    # FIX: Updated to uppercase prefix to match one-hot encoder column formatting
    target_month_col = f'MONTH_{month}'
    target_day_col = f'DAY_OF_WEEK_{day_of_week}'
    target_airline_col = f'AIRLINE_{airline_code}'
    
    if target_month_col in input_data: input_data[target_month_col] = [1.0]
    if target_day_col in input_data: input_data[target_day_col] = [1.0]
    if target_airline_col in input_data: input_data[target_airline_col] = [1.0]
    
    custom_X = pd.DataFrame(input_data)[expected_columns]
    predicted_minutes = model.predict(custom_X)[0]
    
    # ==========================================================
    # 5. GRAPHICAL DISPLAY BANNERS
    # ==========================================================
    st.markdown("---")
    st.subheader("⏱️ Estimated Flight Status Result")
    
    if predicted_minutes > 5:
        st.error(f"⚠️ Our predictive model projects an arrival delay of approximately **{predicted_minutes:.1f} minutes**.")
        st.caption("Consider checking ahead with your carrier for compounding schedule adjustments.")
    elif predicted_minutes < -5:
        st.success(f"🎉 Great news! This exact route configuration historically runs ahead of schedule. Predicted to land early (**{predicted_minutes:.1f} minutes**).")
        st.caption("This flight block is highly optimized by air traffic controllers.")
    else:
        st.info(f"✅ This configuration functions efficiently. Predicted arrival is **On-Time** (Variance: {predicted_minutes:.1f} minutes).")
        st.caption("Your scheduled route showcases strong, stable reliability parameters.")