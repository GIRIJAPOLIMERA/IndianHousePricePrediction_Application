# IndianHousePricePrediction_Application
This project predicts Indian House Prices using Supervised Machine Learning and deploys the model as an interactive web application using Streamlit.
streamlit :  https://indianhousepricepredictionapplication-project-1.streamlit.app/

🏠 Indian House Price Prediction Application

This project is a Machine Learning–based House Price Prediction Web Application built using Python, Scikit-learn, and Streamlit.
It predicts the estimated price (in Lakhs) of residential properties across India based on various property features such as BHK, Size, Property Type, Location, Amenities, and other key housing attributes.

📌 Project Overview

The goal of this project is to help users estimate property prices using a trained Random Forest Regression model.
The application allows users to:

✔ Input property details through an interactive UI
✔ Automatically preprocess categorical & numerical features
✔ Use a trained ML model to predict property price
✔ Apply domain-based post-processing adjustments (e.g., Villa, Independent House pricing factors)
✔ Display the final predicted price in an easy-to-understand format

This makes the app useful for:

Buyers & sellers

Real estate agents

Housing research analysis

Students learning ML + Streamlit

🏗️ Tech Stack
Technology	Purpose
Python	Core programming language
Pandas	Data handling & preprocessing
Scikit-learn	ML pipeline & model training
Streamlit	Interactive web UI
RandomForestRegressor	Price prediction model
📂 Dataset

Name: Indian Housing Prices Dataset

Format: CSV

Source: Kaggle

The dataset contains various attributes of houses in Indian cities and their corresponding prices (in Lakhs).

Target Variable

Price_in_Lakhs

Sample Features Used

Categorical Features:

State

Property Type

Furnished Status

Public Transport Accessibility

Parking Space

Security

Availability Status

Numerical Features:

BHK

Size_in_SqFt

Age of Property

Year Built

Total Floors

Nearby Schools

Nearby Hospitals

⚙️ How the Model Works

Data Preprocessing

Standard scaling for numerical features

One-hot encoding for categorical features

ColumnTransformer + Pipeline used for clean workflow

Model Training

Train-test split (80/20)

RandomForestRegressor used (100 estimators, n_jobs=-1)

Entire pipeline cached for performance

Prediction Logic

User fills property details

Model predicts base price

Domain knowledge rule adjusts price based on property type:

Villa → +25%

Independent House → +15%

Apartment → −5%
