import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 Indian House Price Prediction Application 🏠")
st.write("Prediction of House Price using Machine Learning Model")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("india_housing_prices_small.csv")

data = load_data()

# ----------------------------
# Target and Features
# ----------------------------
target_col = "Price_in_Lakhs"

X = data.drop(columns=[
    target_col,
    "ID",
    "City",
    "Locality",
    "Amenities",
    "Facing",
    "Owner_Type"
])

y = data[target_col]

# ----------------------------
# Column Types
# ----------------------------
categorical_cols = [
    "State",
    "Property_Type",
    "Furnished_Status",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Availability_Status"
]

numerical_cols = [
    "BHK",
    "Size_in_SqFt",
    "Year_Built",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals"
]

# ----------------------------
# Preprocessing
# ----------------------------
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train Model (CACHED)
# ----------------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# ----------------------------
# User Inputs
# ----------------------------
st.subheader("Enter your Requirements")

state = st.selectbox("State", sorted(data["State"].unique()))
property_type = st.selectbox("Property Type", sorted(data["Property_Type"].unique()))
furnished = st.selectbox("Furnished Status", sorted(data["Furnished_Status"].unique()))
transport = st.selectbox("Public Transport Accessibility", ["Low", "Medium", "High"])
parking = st.selectbox("Parking Space", ["Yes", "No"])
security = st.selectbox("Security", ["Yes", "No"])
availability = st.selectbox("Availability Status", sorted(data["Availability_Status"].unique()))

bhk = st.slider("BHK", 1, 6, 2)
size = st.slider("Size (Sq Ft)", 500, 6000, 1200)
year = st.slider("Year Built", 1980, 2024, 2010)
floor = st.slider("Floor No", 0, 30, 2)
total_floors = st.slider("Total Floors", 1, 40, 10)
age = st.slider("Age of Property", 0, 50, 10)
schools = st.slider("Nearby Schools", 0, 20, 5)
hospitals = st.slider("Nearby Hospitals", 0, 20, 3)

# ----------------------------
# Prediction
# ----------------------------
input_data = pd.DataFrame({
    "State": [state],
    "Property_Type": [property_type],
    "Furnished_Status": [furnished],
    "Public_Transport_Accessibility": [transport],
    "Parking_Space": [parking],
    "Security": [security],
    "Availability_Status": [availability],
    "BHK": [bhk],
    "Size_in_SqFt": [size],
    "Year_Built": [year],
    "Floor_No": [floor],
    "Total_Floors": [total_floors],
    "Age_of_Property": [age],
    "Nearby_Schools": [schools],
    "Nearby_Hospitals": [hospitals]
})

if st.button("Predict House Price"):
    base_price = model.predict(input_data)[0]

    # ----------------------------
    # Property Type Adjustment (Domain Knowledge)
    # ----------------------------
    if property_type == "Villa":
        final_price = base_price * 1.25
    elif property_type == "Independent House":
        final_price = base_price * 1.15
    else:  # Apartment
        final_price = base_price * 0.95

    st.success(f"💰 Estimated House Price: ₹ {final_price:.2f} Lakhs")

