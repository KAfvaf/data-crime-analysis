# Data Wrangling - Analisis Niat Kejahatan

Repository ini berisi notebook Jupyter yang berjudul **Data_Wrangling_Niat.ipynb**, yang merupakan bagian dari proses pra-pemodelan dalam analisis data. Fokus utamanya adalah pada **pembersihan dan transformasi data** sebelum dilakukan eksplorasi lebih lanjut atau machine learning.

## 📂 File
- `Data_Wrangling_Niat.ipynb` — Notebook yang berisi proses wrangling dataset, seperti:
  - Konversi waktu (`datetime`)
  - Ekstraksi fitur waktu (jam, hari, bulan, dll)
  - Encoding fitur kategorikal
  - Penanganan nilai kosong
  - Pembuatan fitur baru seperti `TimeOfDay`

## 📊 Fitur yang Diolah
Beberapa kolom penting yang diolah:
- `Dates` → diubah menjadi format `datetime`
- `Hour`, `DayOfWeek`, `Month`, `Year` → diekstraksi dari waktu
- `TimeOfDay` → dikategorikan ke pagi, siang, sore, malam
- `Category`, `PdDistrict`, dll → di-*encode*

## 📌 Tujuan
Proses wrangling ini bertujuan untuk menyiapkan dataset agar siap digunakan pada tahap berikutnya:
- Analisis eksploratif (EDA)
- Visualisasi data
- Modeling (klasifikasi/klasterisasi)

## ⚙️ Tools
Notebook ini menggunakan:
- Python
- Pandas
- NumPy
- Scikit-learn (untuk encoding)

## 🚀 Cara Menjalankan
1. Clone repo ini:
   ```bash
   git clone https://github.com/username/data-wrangling-project.git
   cd data-wrangling-project
