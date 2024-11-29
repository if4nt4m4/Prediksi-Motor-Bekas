import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset for dropdown options
data_path = "motor_second.csv"
data = pd.read_csv(data_path)

# Load the trained model
model = joblib.load("motor_second_model.pkl")

# Extract unique values for dropdowns
model_names = data['model'].unique().tolist()
transmissions = data['transmisi'].unique().tolist()
motor_types = data['jenis'].unique().tolist()

# Define the input form
st.title("Prediksi Harga Sepeda Motor Bekas")
st.write("Masukkan data berikut untuk memprediksi harga sepeda motor bekas:")

# Input fields
model_name = st.selectbox("Model Motor", model_names)
year = st.number_input(
    "Tahun Produksi",
    min_value=int(data['tahun'].min()),
    max_value=int(data['tahun'].max()),
    step=1,
    value=2018
)
transmission = st.selectbox("Transmisi", transmissions)
odometer = st.number_input(
    "Odometer (km)",
    min_value=int(data['odometer'].min()),
    max_value=int(data['odometer'].max()),
    step=1000,
    value=20000
)
motor_type = st.selectbox("Jenis Motor", motor_types)
tax = st.number_input(
    "Pajak (ribuan IDR)",
    min_value=float(data['pajak'].min()),
    max_value=float(data['pajak'].max()),
    step=10.0,
    value=100.0
)
fuel_efficiency = st.number_input(
    "Konsumsi BBM (km/liter)",
    min_value=float(data['konsumsiBBM'].min()),
    max_value=float(data['konsumsiBBM'].max()),
    step=1.0,
    value=50.0
)
engine_capacity = st.number_input(
    "Kapasitas Mesin (cc)",
    min_value=float(data['mesin'].min()),
    max_value=float(data['mesin'].max()),
    step=1.0,
    value=125.0
)

# Predict button
if st.button("Prediksi Harga"):
    # Create input dataframe
    input_data = pd.DataFrame({
        "model": [model_name],
        "tahun": [year],
        "transmisi": [transmission],
        "odometer": [odometer],
        "jenis": [motor_type],
        "pajak": [tax],
        "konsumsiBBM": [fuel_efficiency],
        "mesin": [engine_capacity]
    })
    
    # Predict
    predicted_price = model.predict(input_data)[0]
    st.success(f"Prediksi harga motor bekas: Rp {predicted_price:,.0f}")

# Section: Visualizations
st.title("Analisis Data Sepeda Motor Bekas")
st.write("Berikut adalah visualisasi data untuk membantu analisis:")

# Distribution of prices
st.subheader("Distribusi Harga")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data['harga'], kde=True, bins=20, color='blue', ax=ax)
ax.set_title("Distribusi Harga Sepeda Motor Bekas")
ax.set_xlabel("Harga (Rp)")
ax.set_ylabel("Frekuensi")
st.pyplot(fig)

# Relationship between year and price
st.subheader("Hubungan Tahun Produksi dengan Harga")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=data['tahun'], y=data['harga'], alpha=0.6, ax=ax)
ax.set_title("Tahun Produksi vs Harga")
ax.set_xlabel("Tahun Produksi")
ax.set_ylabel("Harga (Rp)")
st.pyplot(fig)

# Boxplot of model vs price
st.subheader("Distribusi Harga Berdasarkan Model Motor")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='model', y='harga', data=data, ax=ax)
ax.set_title("Harga Berdasarkan Model Motor")
ax.set_xlabel("Model Motor")
ax.set_ylabel("Harga (Rp)")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# Section: Visualizing Prediction Results
st.title("Visualisasi Hasil Prediksi")

# Extract actual and predicted prices for test data
X = data.drop(columns=['harga'])
y = data['harga']

# We will use the entire dataset to make predictions for visualization
predictions = model.predict(X)

# Comparison of actual vs predicted prices
comparison_df = pd.DataFrame({
    "Harga Aktual": y,
    "Harga Prediksi": predictions
})

# Plot actual vs predicted prices
st.subheader("Perbandingan Harga Aktual dan Prediksi")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=comparison_df['Harga Aktual'], y=comparison_df['Harga Prediksi'], alpha=0.6, ax=ax)
ax.set_title("Perbandingan Harga Aktual dan Prediksi")
ax.set_xlabel("Harga Aktual (Rp)")
ax.set_ylabel("Harga Prediksi (Rp)")
st.pyplot(fig)

# Add a line of perfect prediction
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=comparison_df['Harga Aktual'], y=comparison_df['Harga Prediksi'], alpha=0.6, ax=ax)
ax.plot([comparison_df['Harga Aktual'].min(), comparison_df['Harga Aktual'].max()],
        [comparison_df['Harga Aktual'].min(), comparison_df['Harga Aktual'].max()], color='red', linestyle='--')
ax.set_title("Perbandingan Harga Aktual dan Prediksi dengan Garis Sempurna")
ax.set_xlabel("Harga Aktual (Rp)")
ax.set_ylabel("Harga Prediksi (Rp)")
st.pyplot(fig)
