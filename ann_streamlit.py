import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

st.title("Product Demand Prediction (ANN)")

# Upload CSV
data = st.file_uploader("Upload CSV", type="csv")

if data:
    df = pd.read_csv(data)
    st.write("Uploaded dataset:")
    st.write(df.head())

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Check required column
    if 'order_demand' not in df.columns:
        st.error("CSV must have an 'Order_Demand' column!")
        st.stop()

    # Handle date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d', errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df = df.drop('date', axis=1)

    # Fill missing values
    df = df.fillna(0)

    # Keep original categorical columns for prediction UI
    categorical_cols = ['product_code', 'warehouse', 'product_category']

    # Encode categorical columns for training
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Features and target
    X = df_encoded.drop('order_demand', axis=1)
    y = pd.to_numeric(df_encoded['order_demand'], errors='coerce').fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build ANN model
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_scaled.shape[1]),
        Dense(32, activation='relu'),
        Dense(1)  # Regression output
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, y, epochs=50, batch_size=16, verbose=0)

    st.subheader("Make a prediction")

    # Dropdowns for categorical columns
    selected_product = st.selectbox("Product Code", df['product_code'].unique())
    selected_warehouse = st.selectbox("Warehouse", df['warehouse'].unique())
    selected_category = st.selectbox("Product Category", df['product_category'].unique())

    # Numeric inputs for date features
    year = st.number_input("Year", value=2022)
    month = st.number_input("Month", value=1, min_value=1, max_value=12)
    day = st.number_input("Day", value=1, min_value=1, max_value=31)
    weekday = st.number_input("Weekday", value=0, min_value=0, max_value=6)

    # Create single-row DataFrame for prediction
    input_df = pd.DataFrame({
        'year': [year],
        'month': [month],
        'day': [day],
        'weekday': [weekday]
    })

    # Encode selected categories into dummy columns
    for col in categorical_cols:
        dummy_cols = [c for c in X.columns if c.startswith(col + '_')]
        for dummy_col in dummy_cols:
            value = 1 if dummy_col == f"{col}_{locals()['selected_' + col.split('_')[0]]}" else 0
            input_df[dummy_col] = value

    # Ensure all columns match
    input_df = input_df[X.columns]

    # Scale input row
    input_scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(input_scaled)
    st.write(f"Predicted Order Demand: {pred[0][0]:.2f}")

else:
    st.info("Please upload a CSV file.")
