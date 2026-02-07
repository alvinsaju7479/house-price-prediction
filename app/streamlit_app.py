import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "models/model.pkl"

st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Predictor")
st.caption("Random Forest model + preprocessing pipeline (OneHotEncoder + Imputer).")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.sidebar.header("Input Features")

area = st.sidebar.number_input("area", min_value=100, max_value=50000, value=3000, step=50)
bedrooms = st.sidebar.number_input("bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.number_input("bathrooms", min_value=1, max_value=10, value=2)
stories = st.sidebar.number_input("stories", min_value=1, max_value=10, value=2)
parking = st.sidebar.number_input("parking", min_value=0, max_value=10, value=1)

mainroad = st.sidebar.selectbox("mainroad", ["yes", "no"])
guestroom = st.sidebar.selectbox("guestroom", ["yes", "no"])
basement = st.sidebar.selectbox("basement", ["yes", "no"])
hotwaterheating = st.sidebar.selectbox("hotwaterheating", ["yes", "no"])
airconditioning = st.sidebar.selectbox("airconditioning", ["yes", "no"])
prefarea = st.sidebar.selectbox("prefarea", ["yes", "no"])
furnishingstatus = st.sidebar.selectbox("furnishingstatus", ["furnished", "semi-furnished", "unfurnished"])

row = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus
}
X = pd.DataFrame([row])

st.subheader("Your Input")
st.dataframe(X, use_container_width=True)

if st.button("Predict Price ✅"):
    pred = model.predict(X)[0]
    st.success(f"Predicted House Price: **{pred:,.0f}**")

