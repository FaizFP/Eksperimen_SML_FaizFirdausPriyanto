import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# --- KONFIGURASI PATH ---
# Menggunakan os.getcwd() agar path dinamis berdasarkan lokasi run (root repository)
BASE_DIR = os.getcwd()

# Perhatikan nama file input harus sesuai dengan file yang ada di folder raw Anda
# Berdasarkan file yang Anda upload, namanya adalah 'WineQT.csv'
INPUT_FILE = os.path.join(BASE_DIR, 'namadataset_raw', 'WineQT.csv')

# Output path
OUTPUT_DIR = os.path.join(BASE_DIR, 'preprocessing', 'namadataset_preprocessing')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'wine_quality_clean.csv')

def process_data():
    print(f"[INFO] Memulai Preprocessing Otomatis...")
    print(f"[INFO] Mencari data mentah di: {INPUT_FILE}")

    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        # Fallback check: Siapa tahu dijalankan dari dalam folder preprocessing
        if os.path.exists(os.path.join('..', 'namadataset_raw', 'WineQT.csv')):
            INPUT_FILE_ALT = os.path.join('..', 'namadataset_raw', 'WineQT.csv')
            df = pd.read_csv(INPUT_FILE_ALT)
        else:
            raise FileNotFoundError(f"File dataset tidak ditemukan di: {INPUT_FILE}")
    else:
        df = pd.read_csv(INPUT_FILE)
    
    print(f"[INFO] Data berhasil dimuat. Dimensi awal: {df.shape}")

    
    
    # 5. Simpan Hasil ke CSV (PENTING AGAR GITHUB ACTION BISA COMMIT)
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Buat folder output jika belum ada
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Data bersih disimpan di: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()