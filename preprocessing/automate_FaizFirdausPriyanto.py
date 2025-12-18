import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(csv_path, target_col='quality', test_size=0.2, random_state=42):
    """
    Fungsi untuk memuat data yang SUDAH dipreprocessing (sudah bersih & scaling),
    lalu membaginya menjadi set Training dan Testing.

    Parameters:
    csv_path (str): Lokasi file CSV hasil preprocessing dari notebook.
    target_col (str): Nama kolom target (default='quality').
    test_size (float): Proporsi data testing (default=0.2 atau 20%).
    random_state (int): Seed agar hasil split konsisten (default=42).

    Returns:
    X_train, X_test, y_train, y_test
    """
    
    # 1. Load Data dari CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"[INFO] Berhasil memuat data dari: {csv_path}")
        print(f"[INFO] Dimensi data: {df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File tidak ditemukan di path: {csv_path}")

    # 2. Pisahkan Fitur (X) dan Target (y)
    # Kita asumsikan data sudah bersih, jadi langsung drop target saja
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 3. Splitting Data (Train & Test)
    # Kita gunakan stratify=y agar penyebaran kelas di train & test seimbang
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

    print(f"[INFO] Splitting selesai.")
    print(f"      - X_train: {X_train.shape}")
    print(f"      - X_test : {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Contoh penggunaan untuk testing
    # Pastikan nama file sesuai dengan file csv yang sudah Anda simpan di notebook
    try:
        X_tr, X_ts, y_tr, y_ts = load_and_split_data('wine_quality_preprocessed.csv')
        print("\nContoh 5 data fitur training:")
        print(X_tr.head())
    except Exception as e:
        print(f"Error: {e}")