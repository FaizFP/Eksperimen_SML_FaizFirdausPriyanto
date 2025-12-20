import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import os
import pathlib  # <--- Library penting untuk fix path Windows

def train_model():
    # --- 1. Load Data ---
    # Gunakan raw string (r'...') agar aman dari backslash error
    input_path = r'C:\Users\faiz\Documents\tgstgs\SEMESTER 5\Dicoding\Eksperimen_SML_FaizFirdausPriyanto\preprocessing\wine_quality_clean.csv'
    
    if not os.path.exists(input_path):
        print(f"[ERROR] File tidak ditemukan di: {input_path}")
        return

    print(f"[INFO] Memuat data dari: {input_path}")
    df = pd.read_csv(input_path)

    # Pisahkan Fitur (X) dan Target (y)
    target_col = 'quality'
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 2. Setup MLflow (SOLUSI FIX WINDOWS) ---
    # Menggunakan pathlib agar format URI otomatis menjadi file:///C:/... (forward slash)
    # Ini format yang WAJIB dipakai MLflow di Windows
    tracking_uri = pathlib.Path("./mlruns").resolve().as_uri()
    
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"[INFO] MLflow Tracking URI diset ke: {tracking_uri}")
    
    # Set nama eksperimen
    mlflow.set_experiment("Eksperimen_SML_FaizFirdausPriyanto")

    # --- 3. Aktifkan Autolog ---
    mlflow.autolog()

    # --- 4. Training Model ---
    with mlflow.start_run():
        print("[INFO] Memulai training model (RandomForest)...")
        
        # Model tanpa hyperparameter tuning
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        print("[INFO] Training selesai.")
        print(f"[INFO] Cek hasil di folder: {pathlib.Path('./mlruns').resolve()}")

if __name__ == "__main__":
    train_model()