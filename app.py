import streamlit as st
import joblib
import numpy as np

# Load model and scaler
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")

# Custom theme: Improved background and sidebar styling
st.markdown(
    """
    <style>
    body {
        background-color: #e7f3ff; /* Light blue for a calm water theme */
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        text-align: center;
        color: #004aad; /* Darker water blue */
    }
    .stButton>button {
        background-color: #004aad; /* Button color */
        border: none;
        color: white;
        border-radius: 12px;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #003d73;
        color: white;
        transition: 0.3s ease;
    }
    [data-testid="stSidebar"] {
        position: absolute;
        right: 0;
        width: 500px; /* Increased sidebar width */
        background-color: #dbe9f6; /* Light pastel blue for sidebar */
        box-shadow: -2px 0px 5px rgba(0,0,0,0.1);
    }
    .footer {
        background-color: #004aad; /* Blue footer background */
        color: #ffffff; /* Light text for footer */
        text-align: center;
        padding: 20px;
        font-size: 16px;
        margin-top: 30px;
        border-radius: 10px;
    }
    .footer p {
        margin: 0;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.title("🌟 Water Access Predictor By Leslie And Aaron 🌟")
st.subheader("💧 Predict water access levels with meaningful insights! 💧")

# Sidebar inputs for direct features
st.sidebar.header("🌊 Input Key Water Access and Population Metrics")
st.sidebar.markdown(
    """
    📋 **Feature Descriptions:**  
    - **Urban Population (%):** The percentage of the total population living in urban areas.  
    - **Basic Water Access (%):** Percentage of the population with access to clean, safe drinking water.  
    - **Surface Water Access (%):** Percentage of the population relying on untreated surface water sources.  
    """
)
year = st.sidebar.number_input("📅 Year", min_value=2000, max_value=2025, value=2020, step=1)
pop_n = st.sidebar.number_input("👨‍👩‍👧‍👦 Total Population (millions)", min_value=0.0, value=1000.0, step=1.0)
pop_u = st.sidebar.slider("🏙️ Urban Population Percentage", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
wat_bas_n = st.sidebar.slider("🚰 Basic Water Access (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
wat_lim_n = st.sidebar.slider("🚿 Limited Water Access (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
wat_unimp_n = st.sidebar.slider("🪠 Unimproved Water Access (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
wat_sur_n = st.sidebar.slider("🏞️ Surface Water Access (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
wat_bas_r = st.sidebar.slider("🌱 Basic Rural Water Access (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
wat_lim_r = st.sidebar.slider("🌳 Limited Rural Water Access (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
wat_unimp_r = st.sidebar.slider("🌾 Unimproved Rural Water Access (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
wat_bas_u = st.sidebar.slider("🏠 Basic Urban Water Access (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
wat_lim_u = st.sidebar.slider("🏢 Limited Urban Water Access (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
wat_unimp_u = st.sidebar.slider("🏘️ Unimproved Urban Water Access (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
wat_sur_u = st.sidebar.slider("🏚️ Surface Urban Water Access (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)

# Derived features
year_squared = year ** 2
pop_r = 100 - pop_u  # Rural population percentage
pop_unimp_interaction = pop_u * wat_unimp_n
pop_r_lim_interaction = pop_r * wat_lim_n
wat_sur_n_log = np.log1p(wat_sur_n)

# Feature mapping
input_data_dict = {
    'year_squared': year_squared,
    'pop_n': pop_n,
    'pop_u': pop_u,
    'wat_bas_n': wat_bas_n,
    'wat_lim_n': wat_lim_n,
    'wat_unimp_n': wat_unimp_n,
    'wat_sur_n': wat_sur_n,
    'wat_bas_r': wat_bas_r,
    'wat_lim_r': wat_lim_r,
    'wat_unimp_r': wat_unimp_r,
    'wat_sur_r_log': wat_sur_n_log,
    'wat_bas_u': wat_bas_u,
    'wat_lim_u': wat_lim_u,
    'wat_unimp_u': wat_unimp_u,
    'wat_sur_u': wat_sur_u,
    'pop_r_lim_interaction': pop_r_lim_interaction,
    'pop_unimp_interaction': pop_unimp_interaction,
}

# Display dynamic derived features
col1, col2 = st.columns(2)
with col1:
    st.write("🎢 **Derived Features:**")
    st.write(f"1. 📐 Year Squared: {year_squared}")
    st.write(f"2. 🌍 Rural Population (%): {pop_r:.2f}")
    st.write(f"3. 🔗 Urban-Unimproved Interaction: {pop_unimp_interaction:.2f}")
with col2:
    st.write(f"4. 🔗 Rural-Limited Interaction: {pop_r_lim_interaction:.2f}")
    st.write(f"5. 🪞 Log Surface Water Access: {wat_sur_n_log:.2f}")

# Prepare data for prediction
try:
    columns_to_keep = [
        'year_squared', 'pop_n', 'pop_u', 'wat_bas_n', 'wat_lim_n', 'wat_unimp_n', 'wat_sur_n',
        'wat_bas_r', 'wat_lim_r', 'wat_unimp_r', 'wat_sur_r_log',
        'wat_bas_u', 'wat_lim_u', 'wat_unimp_u', 'wat_sur_u',
        'pop_r_lim_interaction', 'pop_unimp_interaction',
    ]
    input_data_list = [input_data_dict.get(col, 0) for col in columns_to_keep]
    input_data_array = np.array([input_data_list])

    # Scale input data
    input_data_scaled = scaler.transform(input_data_array)

    # Prediction button
    if st.button("💧 Predict Now!"):
        prediction = model.predict(input_data_scaled)
        basic_water_access_pct = 100 - prediction[0]  # Calculate basic water access percentage
        total_population = pop_n * 1_000_000  # Convert millions to actual population
        people_with_access = (basic_water_access_pct / 100) * total_population

        st.success(
            f"🌟 Predicted Basic Water Access: {basic_water_access_pct:.2f}%.\n\n"
            f"💧 This means approximately {people_with_access:,.0f} people out of {total_population:,.0f} "
            f"will have access to at least basic water services, including piped water, wells, or clean drinking sources."
        )
        if basic_water_access_pct > 80:
            st.info("👍 Great! Most of the population has access to basic water services.")
        elif basic_water_access_pct < 50:
            st.warning("⚠️ A significant portion of the population lacks access to basic water services.")
        st.balloons()

except Exception as e:
    st.error(f"❌ Error during prediction: {e}")

# Footer explanation
st.markdown(
    """
    <div class="footer">
        <p><strong>What Does This App Do?</strong></p>
        <p>This app predicts the percentage of a population with access to basic water services, using a machine learning model trained on historical water access data. 
        It also estimates the total number of people with access based on the input population size.</p>
        <p>Users input key metrics such as urban and rural population percentages, water access levels (e.g., unimproved, limited, and surface water), and year. 
        Derived features are calculated automatically to capture interactions and trends in the data.</p>
        <p><strong>Use Case:</strong> This tool can help policymakers and researchers understand water access trends, predict future accessibility, and plan interventions to improve water availability where needed.</p>
        <p>Thank you for using the Water Access Predictor! 💧</p>
    </div>
    """,
    unsafe_allow_html=True,
)
