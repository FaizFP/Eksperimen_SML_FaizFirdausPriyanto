import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# --- KONFIGURASI PATH ---
BASE_DIR = os.getcwd()

# Input: Tetap membaca dari folder namadataset_raw
INPUT_FILE = os.path.join(BASE_DIR, 'namadataset_raw', 'WineQT.csv')

# Output: Langsung di folder preprocessing (tidak perlu folder namadataset_preprocessing)
OUTPUT_FILE = os.path.join(BASE_DIR, 'preprocessing', 'wine_quality_clean.csv')

def process_data():
    print(f"[INFO] Memulai Preprocessing Otomatis...")
    print(f"[INFO] Mencari data mentah di: {INPUT_FILE}")

    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        # Fallback check (Jaga-jaga jika dijalankan dari dalam folder preprocessing)
        if os.path.exists(os.path.join('..', 'namadataset_raw', 'WineQT.csv')):
            INPUT_FILE_ALT = os.path.join('..', 'namadataset_raw', 'WineQT.csv')
            df = pd.read_csv(INPUT_FILE_ALT)
        else:
            raise FileNotFoundError(f"File dataset tidak ditemukan di: {INPUT_FILE}")
    else:
        df = pd.read_csv(INPUT_FILE)
    
    print(f"[INFO] Data berhasil dimuat. Dimensi awal: {df.shape}")

    # 2. Data Cleaning (Drop Id jika ada)
    if 'Id' in df.columns:
        df.drop('Id', axis=1, inplace=True)
        print("[INFO] Kolom 'Id' dihapus.")
    
    # 3. Handling Outlier (Metode IQR)
    target_col = 'quality'
    if target_col in df.columns:
        features = df.columns.drop(target_col)
    else:
        features = df.columns
        
    Q1 = df[features].quantile(0.25)
    Q3 = df[features].quantile(0.75)
    IQR = Q3 - Q1
    
    condition = ~((df[features] < (Q1 - 1.5 * IQR)) | (df[features] > (Q3 + 1.5 * IQR))).any(axis=1)
    df_clean = df[condition].copy()
    
    print(f"[INFO] Outlier dibuang: {len(df) - len(df_clean)} baris.")

    # 4. Standarisasi (Scaling)
    scaler = StandardScaler()
    df_clean[features] = scaler.fit_transform(df_clean[features])
    
    # 5. Simpan Hasil
    # Pastikan folder tujuan (preprocessing) ada
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Data bersih disimpan di: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()