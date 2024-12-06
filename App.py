# Mengimpor pustaka yang diperlukan
import streamlit as st  # Untuk membuat antarmuka aplikasi web interaktif
import pandas as pd  # Untuk manipulasi data
import matplotlib.pyplot as plt  # Untuk membuat visualisasi grafik
import seaborn as sns  # Untuk membuat grafik statistik
import joblib  # Untuk memuat model machine learning yang telah dilatih

# Memuat dataset untuk opsi dropdown
data_path = "motor_second.csv"  # Lokasi file dataset
data = pd.read_csv(data_path)  # Membaca dataset ke dalam DataFrame

# Memuat model yang telah dilatih
model = joblib.load("motor_second_model.pkl")  # Memuat model dari file

# Mengambil nilai unik untuk opsi dropdown
model_names = data['model'].unique().tolist()  # Daftar model motor
transmissions = data['transmisi'].unique().tolist()  # Daftar jenis transmisi
motor_types = data['jenis'].unique().tolist()  # Daftar jenis motor

# Membuat form input di aplikasi Streamlit
st.title("Prediksi Harga Sepeda Motor Bekas")  # Judul aplikasi
st.write("metode Supervised Learning (Regresi) dengan model machine learning yang telah dilatih sebelumnya, dimuat dari file .pkl menggunakan pustaka joblib, untuk memprediksi harga sepeda motor bekas berdasarkan fitur input seperti model, tahun produksi, odometer, jenis motor, pajak, konsumsi BBM, dan kapasitas mesin. Input pengguna dikumpulkan melalui antarmuka interaktif yang dibuat dengan Streamlit, dengan validasi nilai minimum dan maksimum berdasarkan dataset yang digunakan. Selain prediksi, kode ini juga menyediakan analisis data menggunakan visualisasi statistik, seperti histogram untuk distribusi harga, scatter plot untuk hubungan antar variabel (misalnya, tahun produksi vs harga atau harga aktual vs harga prediksi), serta box plot untuk distribusi harga berdasarkan model motor. Evaluasi hasil prediksi dilakukan dengan scatter plot yang menampilkan perbandingan harga aktual dan prediksi, dilengkapi garis diagonal untuk menunjukkan prediksi sempurna. Pendekatan ini menggabungkan pengolahan data menggunakan pandas, visualisasi menggunakan matplotlib dan seaborn, serta integrasi model prediktif untuk menghasilkan antarmuka prediksi yang interaktif dan informatif.")
st.write("Masukkan data berikut untuk memprediksi harga sepeda motor bekas:")  # Instruksi

# Input untuk pengguna
model_name = st.selectbox("Model Motor", model_names)  # Dropdown untuk memilih model motor
year = st.number_input(  # Input angka untuk tahun produksi
    "Tahun Produksi",
    min_value=int(data['tahun'].min()),  # Tahun minimum
    max_value=int(data['tahun'].max()),  # Tahun maksimum
    step=1,  # Langkah input
    value=2018  # Nilai default
)
transmission = st.selectbox("Transmisi", transmissions)  # Dropdown untuk transmisi
odometer = st.number_input(  # Input angka untuk odometer
    "Odometer (km)",
    min_value=int(data['odometer'].min()),  # Nilai minimum
    max_value=int(data['odometer'].max()),  # Nilai maksimum
    step=1000,  # Langkah input
    value=20000  # Nilai default
)
motor_type = st.selectbox("Jenis Motor", motor_types)  # Dropdown untuk jenis motor
tax = st.number_input(  # Input angka untuk pajak
    "Pajak (ribuan IDR)",
    min_value=float(data['pajak'].min()),  # Nilai minimum
    max_value=float(data['pajak'].max()),  # Nilai maksimum
    step=10.0,  # Langkah input
    value=100.0  # Nilai default
)
fuel_efficiency = st.number_input(  # Input angka untuk konsumsi BBM
    "Konsumsi BBM (km/liter)",
    min_value=float(data['konsumsiBBM'].min()),  # Nilai minimum
    max_value=float(data['konsumsiBBM'].max()),  # Nilai maksimum
    step=1.0,  # Langkah input
    value=50.0  # Nilai default
)
engine_capacity = st.number_input(  # Input angka untuk kapasitas mesin
    "Kapasitas Mesin (cc)",
    min_value=float(data['mesin'].min()),  # Nilai minimum
    max_value=float(data['mesin'].max()),  # Nilai maksimum
    step=1.0,  # Langkah input
    value=125.0  # Nilai default
)

# Tombol untuk memprediksi
if st.button("Prediksi Harga"):
    # Membuat DataFrame dari input pengguna
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
    
    # Melakukan prediksi menggunakan model
    predicted_price = model.predict(input_data)[0]
    st.success(f"Prediksi harga motor bekas: Rp {predicted_price:,.0f}")  # Menampilkan hasil prediksi

# Visualisasi data
st.title("Analisis Data Sepeda Motor Bekas")  # Judul untuk visualisasi
st.write("Berikut adalah visualisasi data untuk membantu analisis:")  # Penjelasan

# Distribusi harga
st.subheader("Distribusi Harga")  # Subjudul
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data['harga'], kde=True, bins=20, color='blue', ax=ax)  # Histogram harga
ax.set_title("Distribusi Harga Sepeda Motor Bekas")
ax.set_xlabel("Harga (Rp)")
ax.set_ylabel("Frekuensi")
st.pyplot(fig)

# Hubungan tahun produksi dengan harga
st.subheader("Hubungan Tahun Produksi dengan Harga")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=data['tahun'], y=data['harga'], alpha=0.6, ax=ax)  # Scatter plot
ax.set_title("Tahun Produksi vs Harga")
ax.set_xlabel("Tahun Produksi")
ax.set_ylabel("Harga (Rp)")
st.pyplot(fig)

# Distribusi harga berdasarkan model motor
st.subheader("Distribusi Harga Berdasarkan Model Motor")
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='model', y='harga', data=data, ax=ax)  # Boxplot harga per model
ax.set_title("Harga Berdasarkan Model Motor")
ax.set_xlabel("Model Motor")
ax.set_ylabel("Harga (Rp)")
ax.tick_params(axis='x', rotation=45)  # Rotasi label x
st.pyplot(fig)

# Visualisasi hasil prediksi
st.title("Visualisasi Hasil Prediksi")  # Judul bagian visualisasi prediksi

# Membandingkan harga aktual dengan harga prediksi
X = data.drop(columns=['harga'])  # Fitur input
y = data['harga']  # Target output

# Prediksi seluruh dataset untuk visualisasi
predictions = model.predict(X)

# DataFrame untuk membandingkan harga aktual dan prediksi
comparison_df = pd.DataFrame({
    "Harga Aktual": y,
    "Harga Prediksi": predictions
})

# Scatter plot harga aktual vs prediksi
st.subheader("Perbandingan Harga Aktual dan Prediksi")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=comparison_df['Harga Aktual'], y=comparison_df['Harga Prediksi'], alpha=0.6, ax=ax)
ax.set_title("Perbandingan Harga Aktual dan Prediksi")
ax.set_xlabel("Harga Aktual (Rp)")
ax.set_ylabel("Harga Prediksi (Rp)")
st.pyplot(fig)

# Scatter plot dengan garis sempurna prediksi
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=comparison_df['Harga Aktual'], y=comparison_df['Harga Prediksi'], alpha=0.6, ax=ax)
ax.plot([comparison_df['Harga Aktual'].min(), comparison_df['Harga Aktual'].max()],
        [comparison_df['Harga Aktual'].min(), comparison_df['Harga Aktual'].max()], color='red', linestyle='--')
ax.set_title("Perbandingan Harga Aktual dan Prediksi dengan Garis Sempurna")
ax.set_xlabel("Harga Aktual (Rp)")
ax.set_ylabel("Harga Prediksi (Rp)")
st.pyplot(fig)