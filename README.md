import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Baca Dataset
df = pd.read_csv('test.csv')
display(df.head(10))

# Salin DataFrame untuk bekerja, menjaga originalitas
df_wrangled = df.copy()

#  Pemahaman Data
print("\n--- Tahap 1: Discovery (Eksplorasi Data Awal) ---")
print("\nBentuk dataset (baris, kolom):", df_wrangled.shape)
print("\nStatistik deskriptif (numerik):")
print(df_wrangled.describe())
print("\nStatistik deskriptif (kategorikal/objek):")
print(df_wrangled.describe(include='object'))

# Cek nilai hilang
print("\nJumlah nilai hilang per kolom:")
print(df_wrangled.isnull().sum())

# Cek duplikasi baris
print("\nJumlah baris duplikat:", df_wrangled.duplicated().sum())

# Distribusi 'DayOfWeek' dan 'PdDistrict'
print("\nDistribusi 'DayOfWeek':")
print(df_wrangled['DayOfWeek'].value_counts(normalize=True))
print("\nDistribusi 'PdDistrict':")
print(df_wrangled['PdDistrict'].value_counts(normalize=True))

# a. Menangani Data Duplikat (jika ada)
if df_wrangled.duplicated().sum() > 0:
    print(f"Menghapus {df_wrangled.duplicated().sum()} baris duplikat.")
    df_wrangled.drop_duplicates(inplace=True)
    print("Jumlah baris setelah menghapus duplikat:", df_wrangled.shape[0])
else:
    print("Tidak ada baris duplikat ditemukan.")

# b. Menangani Nilai Hilang (Missing Values)
# Pada dataset ini, missing values jarang terjadi kecuali pada X, Y yang bisa berupa outlier yang salah.
# Jika ada missing values pada 'Category', 'DayOfWeek', 'PdDistrict', dll., ini harus ditangani.
# Untuk X dan Y, seringkali ada nilai -120.5 atau nilai yang tidak masuk akal.
# Mengganti nilai koordinat yang sangat tidak masuk akal dengan NaN agar bisa diimputasi
df_wrangled['X'] = df_wrangled['X'].replace(-120.5, np.nan) # Ini nilai umum yang salah untuk San Francisco
# Periksa apakah ada nilai Y yang tidak masuk akal (misal Y=90.0)
df_wrangled['Y'] = df_wrangled['Y'].replace(90.0, np.nan)


# Imputasi nilai hilang pada X dan Y dengan median (lebih robust terhadap outlier)
if df_wrangled['X'].isnull().any() or df_wrangled['Y'].isnull().any():
    print("Imputasi nilai hilang pada 'X' dan 'Y' dengan median.")
    imputer_coords = SimpleImputer(strategy='median')
    df_wrangled[['X', 'Y']] = imputer_coords.fit_transform(df_wrangled[['X', 'Y']])
else:
    print("Tidak ada nilai hilang yang perlu diimputasi pada 'X' dan 'Y'.")

print("\nJumlah nilai hilang setelah pembersihan (X, Y):")
print(df_wrangled.isnull().sum())

# c. Penanganan Outlier pada Koordinat X dan Y (Winsorization/Capping)
# Meskipun sudah diimputasi, ada baiknya melakukan capping untuk menjaga X dan Y dalam rentang realistis San Francisco.
# Rentang koordinat yang wajar untuk San Francisco:
# X (Longitude): sekitar -122.51 hingga -122.35
# Y (Latitude): sekitar 37.70 hingga 37.82
# Kita akan menggunakan persentil sebagai batas.
for col in ['X', 'Y']:
    lower_bound = df_wrangled[col].quantile(0.005) # Persentil 0.5%
    upper_bound = df_wrangled[col].quantile(0.995) # Persentil 99.5%
    df_wrangled[col] = np.clip(df_wrangled[col], lower_bound, upper_bound)
    print(f"Kolom '{col}' di-capping antara {lower_bound:.4f} dan {upper_bound:.4f}.")

def get_time_of_day(hour):
    if 0 <= hour < 4:
        return 'Late Night'
    elif 4 <= hour < 8:
        return 'Dawn'
    elif 8 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 16:
        return 'Afternoon'
    elif 16 <= hour < 20:
        return 'Evening'
    else:
        return 'Night'

df_wrangled['TimeOfDay'] = df_wrangled['Hour'].apply(get_time_of_day)

print(df_wrangled.isnull().sum())
print(df_wrangled['TimeOfDay'].value_counts())

df_wrangled['IsWeekend'] = df_wrangled['DayOfWeek_Num'].isin([5, 6]).astype(int)

df_wrangled = df_wrangled.drop_duplicates()

# Convert 'Dates' to datetime objects
df_wrangled['Dates'] = pd.to_datetime(df_wrangled['Dates'])

# a. Ekstraksi Fitur Waktu dari 'Dates'
df_wrangled['Year'] = df_wrangled['Dates'].dt.year
df_wrangled['Month'] = df_wrangled['Dates'].dt.month
df_wrangled['Day'] = df_wrangled['Dates'].dt.day
df_wrangled['Hour'] = df_wrangled['Dates'].dt.hour
df_wrangled['Minute'] = df_wrangled['Dates'].dt.minute
df_wrangled['DayOfWeek_Num'] = df_wrangled['Dates'].dt.dayofweek # Monday=0, Sunday=6
df_wrangled['WeekOfYear'] = df_wrangled['Dates'].dt.isocalendar().week.astype(int) # Minggu ke berapa dalam setahun
df_wrangled['TimeOfDay'] = (df_wrangled['Hour'] % 24 + 4) // 4 # Pagi, Siang, Sore, Malam (0-5: Malam, 6-11: Pagi, 12-17: Siang, 18-23: Sore)
df_wrangled['TimeOfDay'] = df_wrangled['TimeOfDay'].map({
    1: 'Dawn', 2: 'Morning', 3: 'Afternoon', 4: 'Evening', 5: 'Night', 0: 'Late Night'
}) # Mengubah angka ke label

print("\nFitur waktu baru ditambahkan:")
print(df_wrangled[['Dates', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek_Num', 'WeekOfYear', 'TimeOfDay']].head())

def get_time_of_day(hour):
    if 0 <= hour < 4:
        return 'Late Night'
    elif 4 <= hour < 8:
        return 'Dawn'
    elif 8 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 16:
        return 'Afternoon'
    elif 16 <= hour < 20:
        return 'Evening'
    else:
        return 'Night'

df_wrangled['TimeOfDay'] = df_wrangled['Hour'].apply(get_time_of_day)

# b. Rekayasa Fitur dari 'Address' (Sederhana)
# Mengidentifikasi apakah alamat mengandung 'Block' atau 'Street' atau 'Hwy' atau 'Intersection'
df_wrangled['Is_Block'] = df_wrangled['Address'].str.contains('Block', na=False, case=False).astype(int)
df_wrangled['Is_Intersection'] = df_wrangled['Address'].str.contains('/', na=False).astype(int) # '/' seringkali menunjukkan persimpangan
df_wrangled['Is_Hwy'] = df_wrangled['Address'].str.contains('Hwy', na=False, case=False).astype(int)
print("\nFitur 'Address' baru ditambahkan (Is_Block, Is_Intersection, Is_Hwy):")
print(df_wrangled[['Address', 'Is_Block', 'Is_Intersection', 'Is_Hwy']].head())


# c. Drop kolom yang tidak relevan atau sudah diekstrak informasinya
# 'Dates': Sudah diekstrak
# 'Id': ID unik, tidak relevan untuk prediksi kategori
# 'Address': Sudah direkayasa fiturnya, string aslinya terlalu banyak unik untuk encoding.
columns_to_drop = ['Dates', 'Id', 'Address']
# Check if columns exist before dropping
existing_columns_to_drop = [col for col in columns_to_drop if col in df_wrangled.columns]
df_processed = df_wrangled.drop(columns=existing_columns_to_drop)
print(f"\nKolom {existing_columns_to_drop} telah di-drop.")

# Contoh dataframe df_wrangled, pastikan kolom 'Dates' sudah ada
df_wrangled['Dates'] = pd.to_datetime(df_wrangled['Dates'])

# Ekstraksi fitur waktu
df_wrangled['Year'] = df_wrangled['Dates'].dt.year
df_wrangled['Month'] = df_wrangled['Dates'].dt.month
df_wrangled['Day'] = df_wrangled['Dates'].dt.day
df_wrangled['Hour'] = df_wrangled['Dates'].dt.hour
df_wrangled['Minute'] = df_wrangled['Dates'].dt.minute
df_wrangled['DayOfWeek_Num'] = df_wrangled['Dates'].dt.dayofweek
df_wrangled['WeekOfYear'] = df_wrangled['Dates'].dt.isocalendar().week.astype(int)

# Fungsi kategori waktu dari jam
def get_time_of_day(hour):
    if 0 <= hour < 4:
        return 'Late Night'
    elif 4 <= hour < 8:
        return 'Dawn'
    elif 8 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 16:
        return 'Afternoon'
    elif 16 <= hour < 20:
        return 'Evening'
    else:
        return 'Night'

df_wrangled['TimeOfDay'] = df_wrangled['Hour'].apply(get_time_of_day)

# Label Encoder
label_encoder = LabelEncoder()
df_wrangled['TimeOfDay_LabelEncoded'] = label_encoder.fit_transform(df_wrangled['TimeOfDay'])

# One Hot Encoder
one_hot = pd.get_dummies(df_wrangled['TimeOfDay'], prefix='TimeOfDay')
df_encoded = pd.concat([df_wrangled, one_hot], axis=1)

print("Contoh hasil setelah encoding:")
print(df_encoded[['TimeOfDay', 'TimeOfDay_LabelEncoded'] + list(one_hot.columns)])

# b. Encoding Fitur Kategorikal (One-Hot Encoding)
# Kolom yang akan di-One-Hot Encode: 'DayOfWeek', 'PdDistrict', 'TimeOfDay'
ohe_cols = ['DayOfWeek', 'PdDistrict', 'TimeOfDay']

# Menggunakan OneHotEncoder dari sklearn.preprocessing
# Ini akan membuat array, jadi kita akan menggabungkannya kembali ke DataFrame
encoder_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # handle_unknown='ignore' untuk data test
encoded_features = encoder_ohe.fit_transform(df_encoded[ohe_cols])

# Mendapatkan nama kolom baru setelah One-Hot Encoding
encoded_feature_names = encoder_ohe.get_feature_names_out(ohe_cols)

# Membuat DataFrame dari fitur yang di-encode
df_ohe_features = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_encoded.index)

# Menggabungkan kembali dengan DataFrame asli dan menghapus kolom asli yang di-encode
df_final = pd.concat([df_encoded.drop(columns=ohe_cols), df_ohe_features], axis=1)
print(f"\nKolom setelah One-Hot Encoding untuk {ohe_cols}:")
print(df_final.columns)
print(f"Bentuk setelah One-Hot Encoding: {df_final.shape}")

# c. Scaling Fitur Numerik (Untuk Persiapan Model ML)
# Fitur numerik yang perlu di-scaling: X, Y, Year, Month, Day, Hour, Minute, DayOfWeek_Num, WeekOfYear
numeric_features_to_scale = [
    'X', 'Y', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek_Num', 'WeekOfYear',
    'Is_Block', 'Is_Intersection', 'Is_Hwy' # Ini juga numerik biner, scaling tidak akan mengubahnya banyak tapi bisa disertakan
]

# Ambil hanya kolom numerik yang ada di DataFrame dan yang bukan target
actual_numeric_cols_for_scaling = [col for col in numeric_features_to_scale if col in df_final.columns and col != 'Category_Encoded']

if actual_numeric_cols_for_scaling:
    print(f"\nMelakukan Scaling (StandardScaler) pada kolom: {actual_numeric_cols_for_scaling}")
    scaler = StandardScaler()
    df_final[actual_numeric_cols_for_scaling] = scaler.fit_transform(df_final[actual_numeric_cols_for_scaling])
    print("Kolom numerik telah di-scaling.")
else:
    print("\nTidak ada kolom numerik yang teridentifikasi untuk Scaling (atau sudah di-drop).")

print("\n--- Ringkasan Dataset Akhir ---")
print(df_final.head())
print("\nInformasi dataset akhir:")
df_final.info()
print("\nJumlah nilai hilang terakhir:", df_final.isnull().sum().sum())


# --- 5. Validating (Validasi Data Akhir) ---
print("\n--- Tahap 5: Validasi Akhir Data ---")
# Verifikasi tidak ada nilai NaN
if df_final.isnull().sum().sum() == 0:
    print("Validasi: Tidak ada nilai hilang ditemukan di DataFrame akhir.")
else:
    print("Validasi: PERINGATAN! Masih ada nilai hilang di DataFrame akhir.")
    print(df_final.isnull().sum()[df_final.isnull().sum() > 0]) # Tampilkan kolom dengan nilai hilang

# Verifikasi tipe data
print("\nValidasi: Tipe data akhir DataFrame:")
print(df_final.dtypes.value_counts())
if len(df_final.select_dtypes(include=['object']).columns) == 0:
    print("Validasi: Semua kolom fitur numerik (atau biner dari OHE).")
else:
    print(f"Validasi: PERINGATAN! Masih ada kolom non-numerik: {df_final.select_dtypes(include=['object']).columns.tolist()}")

# Cek rentang nilai setelah scaling
print("\nValidasi: Statistik deskriptif setelah scaling (untuk beberapa kolom):")
print(df_final[['X', 'Y', 'Hour', 'Year']].describe()) # Contoh kolom yang di-scaling

# Menyimpan DataFrame yang telah dibersihkan dan ditransformasi
output_filename = 'prepared_sf_crime_data.csv'
df_final.to_csv(output_filename, index=False)
print(f"\nData yang siap disajikan telah disimpan ke '{output_filename}'")
print("Anda juga bisa menyimpan ke format lain seperti Parquet untuk performa lebih baik:")
# df_final.to_parquet('prepared_sf_crime_data.parquet', index=False)

# Visualisasi Distribusi Kejahatan per Jam
plt.figure(figsize=(10, 6))
sns.histplot(df_wrangled['Hour'], bins=24, kde=False, stat='count') # Menggunakan df_wrangled untuk Hour asli
plt.title('Distribusi Kejahatan Berdasarkan Jam dalam Sehari')
plt.xlabel('Jam')
plt.ylabel('Jumlah Kejahatan')
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Visualisasi Kejahatan per Hari dalam Seminggu
plt.figure(figsize=(10, 6))
sns.countplot(x='DayOfWeek', data=df_wrangled, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], palette='viridis')
plt.title('Jumlah Kejahatan Berdasarkan Hari dalam Seminggu')
plt.xlabel('Hari dalam Seminggu')
plt.ylabel('Jumlah Kejahatan')
plt.show()

# Visualisasi Jumlah Kejahatan per Distrik Polisi
plt.figure(figsize=(12, 7))
sns.countplot(y='PdDistrict', data=df_wrangled, order=df_wrangled['PdDistrict'].value_counts().index, palette='rocket')
plt.title('Jumlah Kejahatan per Distrik Polisi')
plt.xlabel('Jumlah Kejahatan')
plt.ylabel('Distrik Polisi')
plt.show()

# Scatter plot Lokasi Kejahatan (sampel kecil untuk kinerja)
# X dan Y setelah capping, jadi outlier yang ekstrem sudah dihilangkan.
plt.figure(figsize=(10, 10))
# Ambil sampel kecil untuk visualisasi jika dataset sangat besar
sample_df = df_final.sample(n=50000, random_state=42) if len(df_final) > 100000 else df_final
sns.scatterplot(x='X', y='Y', data=sample_df, alpha=0.1, s=5, edgecolor=None)
plt.title('Sebaran Geografis Kejahatan di San Francisco (Sampel)')
plt.xlabel('Longitude (Scaled)')
plt.ylabel('Latitude (Scaled)')
plt.show()

print("\n--- Data Wrangling Lengkap dan Siap Disajikan untuk Klasifikasi Kejahatan! ---")

