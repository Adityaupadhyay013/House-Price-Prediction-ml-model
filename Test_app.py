import streamlit as st
import pandas as pd
import joblib
# -----------------------------
# Page Configuration
# -----------------------------
def Outliers_Handling(df):
    df['bedrooms'] = df['bedrooms'].clip(lower = 0.6743252186399298 , upper = 6.127413911794853)
    df['bathrooms'] = df['bathrooms'].clip(lower = 0.625 , upper = 3.625)
    df['sqft_living'] = df['sqft_living'].clip(lower =-280 , upper = 4360)
    df['sqft_lot'] = df['sqft_lot'].clip(lower =-4000 , upper = 20002)
    df['floors'] = df['floors'].clip(lower =-0.5 , upper = 3.5)
    df['sqft_above'] = df['sqft_above'].clip(lower = -475 , upper = 3965)
    df['sqft_basement'] = df['sqft_basement'].clip(lower = -915 , upper = 1525)
    return df
def show_outlier_warnings(inputs):
    warnings = []

    if inputs["bedrooms"][0] < 0.6743252186399298:
        warnings.append(
            "Number of bedrooms is way less than expected. "
            "Input will be capped to 0.67432."
        )

    if inputs["bedrooms"][0] > 6.127413911794853:
        warnings.append(
            "Number of bedrooms is way more than expected. "
            "Input will be capped to 6.12741."
        )

    if inputs["bathrooms"][0] < 0.625:
        warnings.append(
            "Number of bathrooms is way less than expected. "
            "Input will be capped to 0.625."
        )

    if inputs["bathrooms"][0] > 3.625:
        warnings.append(
            "Number of bathrooms is way more than expected. "
            "Input will be capped to 3.625."
        )

    if inputs["sqft_living"][0] > 4360:
        warnings.append(
            "Living area is way more than expected. "
            "Input will be capped to 4360."
        )

    if inputs["sqft_lot"][0] > 20002:
        warnings.append(
            "Lot area is way more than expected. "
            "Input will be capped to 20002."
        )

    if inputs["floors"][0] > 3.5:
        warnings.append(
            "Number of floors is way more than expected. "
            "Input will be capped to 3.5."
        )

    if inputs["sqft_above"][0] > 3965:
        warnings.append(
            "Ceiling area is way more than expected. "
            "Input will be capped to 3965."
        )

    if inputs["sqft_basement"][0] > 1525:
        warnings.append(
            "Basement area is way more than expected. "
            "Input will be capped to 1525."
        )

    return warnings

st.set_page_config(
    page_title="🏠 House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load(r"c:\Users\harsh\House Price Prediction Model(Final).joblib")
# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    color: #1f2937;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stButton>button:hover {
    background-color: #1d4ed8;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title Section
# -----------------------------
st.markdown("<h1 style='text-align: center;'>🏠 House Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict house prices using machine learning</p>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# Input Section
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📌 Enter House Details")

col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input("Bedrooms", 0, 20, 3)
    bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 2.0)
    sqft_living = st.number_input("Sqft Living", 100, 10000, 1800)
    sqft_lot = st.number_input("Sqft Lot", 100, 50000, 4000)

with col2:
    floors = st.number_input("Floors", 1.0, 5.0, 1.0)
    waterfront = st.selectbox("Waterfront", [0, 1])
    view = st.slider("View Rating", 0, 4, 0)
    condition = st.slider("Condition", 1, 5, 3)

with col3:
    sqft_above = st.number_input("Sqft Above", 100, 10000, 1500)
    sqft_basement = st.number_input("Sqft Basement", 0, 5000, 0)
    yr_built = st.number_input("Year Built", 1900, 2025, 2005)
    yr_renovated = st.number_input("Year Renovated", 1900, 2025, 2005)
    zip_code = st.number_input("ZIP Code", 10000, 99999, 98103)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
st.write("")
if st.button("🔮 Predict House Price"):
    features = pd.DataFrame({
    "bedrooms": [bedrooms] , 
    "bathrooms": [bathrooms] , 
    "sqft_living": [sqft_living] , 
    "sqft_lot":	[sqft_lot] ,
    "floors": [floors] , 
    "waterfront": [waterfront] , 
    "view": [view] , 
    "condition": [condition] , 
    "sqft_above": [sqft_above] , 
    "sqft_basement": [sqft_basement] , 
    "yr_built": [yr_built] , 
    "yr_renovated": [yr_renovated] , 
    "zip": [zip_code]
    })
    features = Outliers_Handling(features)
    prediction = model.predict(features)[0]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💰 Estimated House Price")
    st.success(f" {prediction:,.2f} $$")
    st.markdown("</div>", unsafe_allow_html=True)
    warnings = show_outlier_warnings(features)
    if warnings:
     st.warning("⚠️ Input Adjustment Notice")
     for msg in warnings:
        st.write("•", msg)


# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr>
<p style="text-align:center;">
Built with ❤️ using Streamlit & Machine Learning
</p>
""", unsafe_allow_html=True)
