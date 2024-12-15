import streamlit as st
import joblib
import numpy as np

# Load model and scaler
try:
    model = joblib.load('model3.pkl')
    scaler = joblib.load('scaler3.pkl')
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")

st.markdown(
    """
    <style>
    /* General body styling */
    body {
        background-color: #f0f8ff; /* Alice blue */
        font-family: 'Roboto', sans-serif;
    }

    /* Headings */
    h1, h2, h3 {
        text-align: center;
        color: #0056a3; /* Deep blue */
    }

    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #1f2937 !important; /* Dark slate gray */
        color: #ffffff !important; /* White text */
    }

    /* Sidebar Headings */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f9fafb !important; 
        font-size: 18px !important;
    }

    /* Radio Group - Fixing All Labels */
    div[data-baseweb="radio"] {
        color: #ffffff !important; /* Force white text */
    }
    div[data-baseweb="radio"] span {
        color: #ffffff !important; /* Ensure all options are white */
        font-size: 15px !important;
    }

    /* Selected Radio Option Highlight */
    div[data-baseweb="radio"] [aria-checked="true"] span {
        color: #90cdf4 !important; /* Light blue for active */
        font-weight: bold !important;
    }

    /* Slider Labels and Values */
    .stSlider label, .stSlider div {
        color: #f9fafb !important; /* White for sliders */
    }

    /* Sidebar Markdown Text (Fix for Text like 'Input Type') */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label {
        color: #ffffff !important; /* White text */
        font-size: 15px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation bar
st.sidebar.title("Navigation")
nav_selection = st.sidebar.radio("Go to", ["Predict", "About"])

# About Page
if nav_selection == "About":
    st.title("🌟 About the Surplus Water Predictor")
    st.markdown(
        """
        ## About the App
        The **Surplus Water Predictor** estimates **Surplus Water Usage (%)**, 
        which reflects water consumption beyond sustainable limits. This highlights areas 
        where water management strategies are needed to minimize waste and improve efficiency.

        ### Features
        - Predict surplus water usage for **future years (2025 onward)**.
        - Flexible input options: sliders for exploration or manual inputs.
        - Provides actionable insights to address overconsumption.

        ### Use Cases
        - **Policy-making**: Inform policies to reduce water wastage.
        - **Infrastructure Planning**: Identify areas needing improved water infrastructure.
        - **Public Awareness**: Highlight the importance of sustainable water usage.

        Thank you for using the **Surplus Water Predictor**! 💧
        """
    )
    # st.image("https://images.unsplash.com/photo-1602993896853-1d47a1c9d5c7", caption="Water Sustainability", use_column_width=True)

# Prediction Page
if nav_selection == "Predict":
    # App Title and Subheader
    st.title("🌟 Surplus Water Predictor By Leslie And Aaron 🌟")
    st.subheader("✨ Predict surplus water usage for future years! ✨")

    # Sidebar inputs
    st.sidebar.header("🌊 Input Your Features")
    input_type = st.sidebar.radio("Input Type", ["Use Sliders", "Enter Manually"], index=0)

    if input_type == "Use Sliders":
        year = st.sidebar.slider("📅 Year (2025 onward)", min_value=2025, max_value=2050, value=2025, step=1)
        pop_n = st.sidebar.slider("👨‍👩‍👧‍👦 Total Population (millions)", min_value=0.0, max_value=5000.0, value=1000.0, step=10.0)
        pop_u = st.sidebar.slider("🏙️ Urban Population Percentage", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        wat_bas_n = st.sidebar.slider("🚰 Basic Water Access (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        wat_lim_n = st.sidebar.slider("🚿 Limited Water Access (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        wat_unimp_n = st.sidebar.slider("🪠 Unimproved Water Access (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        wat_bas_r = st.sidebar.slider("🌱 Basic Rural Water Access (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        wat_lim_r = st.sidebar.slider("🌳 Limited Rural Water Access (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        wat_unimp_r = st.sidebar.slider("🌾 Unimproved Rural Water Access (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
        wat_bas_u = st.sidebar.slider("🏠 Basic Urban Water Access (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
        wat_lim_u = st.sidebar.slider("🏢 Limited Urban Water Access (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        wat_unimp_u = st.sidebar.slider("🏘️ Unimproved Urban Water Access (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
    else:
        year = st.sidebar.number_input("📅 Year (2025 onward)", min_value=2025, value=2025, step=1)
        pop_n = st.sidebar.number_input("👨‍👩‍👧‍👦 Total Population (millions)", min_value=0.0, value=1000.0, step=1.0)
        pop_u = st.sidebar.number_input("🏙️ Urban Population Percentage", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        wat_bas_n = st.sidebar.number_input("🚰 Basic Water Access (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        wat_lim_n = st.sidebar.number_input("🚿 Limited Water Access (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        wat_unimp_n = st.sidebar.number_input("🪠 Unimproved Water Access (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        wat_bas_r = st.sidebar.number_input("🌱 Basic Rural Water Access (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        wat_lim_r = st.sidebar.number_input("🌳 Limited Rural Water Access (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        wat_unimp_r = st.sidebar.number_input("🌾 Unimproved Rural Water Access (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
        wat_bas_u = st.sidebar.number_input("🏠 Basic Urban Water Access (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
        wat_lim_u = st.sidebar.number_input("🏢 Limited Urban Water Access (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
        wat_unimp_u = st.sidebar.number_input("🏘️ Unimproved Urban Water Access (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)

    # Derived features
    year_squared = year ** 2
    pop_r = 100 - pop_u  # Rural population percentage
    pop_unimp_interaction = pop_u * wat_unimp_n
    pop_r_lim_interaction = pop_r * wat_lim_n

    # Feature mapping
    input_data_dict = {
        'year_squared': year_squared,
        'pop_n': pop_n,
        'pop_u': pop_u,
        'wat_bas_n': wat_bas_n,
        'wat_lim_n': wat_lim_n,
        'wat_unimp_n': wat_unimp_n,
        'wat_bas_r': wat_bas_r,
        'wat_lim_r': wat_lim_r,
        'wat_unimp_r': wat_unimp_r,
        'wat_bas_u': wat_bas_u,
        'wat_lim_u': wat_lim_u,
        'wat_unimp_u': wat_unimp_u,
        'pop_r_lim_interaction': pop_r_lim_interaction,
        'pop_unimp_interaction': pop_unimp_interaction,
        'wat_sur_r_log': 0,
        'wat_sur_u': 0,
    }

    # Prepare data for prediction
    try:
        columns_to_keep = [
            'year_squared', 'pop_n', 'pop_u', 'wat_bas_n', 'wat_lim_n', 'wat_unimp_n',
            'wat_bas_r', 'wat_lim_r', 'wat_unimp_r',
            'wat_bas_u', 'wat_lim_u', 'wat_unimp_u',
            'pop_r_lim_interaction', 'pop_unimp_interaction',
            'wat_sur_r_log', 'wat_sur_u'
        ]

        input_data_list = [input_data_dict.get(col, 0) for col in columns_to_keep]
        input_data_array = np.array([input_data_list])

        # Scale input data
        input_data_scaled = scaler.transform(input_data_array)

        if st.button("✨ Predict Now! ✨"):
            prediction = model.predict(input_data_scaled)
            surplus_water_usage_pct = prediction[0]
            total_population = pop_n * 1_000_000  # Convert to actual number
            people_relying_on_surplus_water = (surplus_water_usage_pct / 100) * total_population

            st.success(
                f"🌟 Predicted Surplus Water Usage in {year}: {surplus_water_usage_pct:.2f}%.\n\n"
                f"Approximately {people_relying_on_surplus_water:,.0f} people out of {total_population:,.0f} "
                f"may consume water beyond sustainable limits."
            )
            st.info(
                f"💡 The prediction accounts for non-linear time trends via the **year squared** feature "
                f"({year_squared}), enabling robust forecasting."
            )
            st.balloons()

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
