import pandas as pd
import mlflow
import mlflow.sklearn
import os
import pathlib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_with_tuning():
    # --- 1. Load Data ---
    # Menggunakan path absolute yang aman untuk Windows
    # Pastikan path ini mengarah ke file hasil preprocessing Anda
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

    # --- 2. Setup MLflow ---
    # Gunakan pathlib agar format URI valid di Windows (file:///C:/...)
    tracking_uri = pathlib.Path("./mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set nama eksperimen (Bisa sama atau beda dengan sebelumnya)
    experiment_name = "Eksperimen_SML_FaizFirdausPriyanto_Tuning"
    mlflow.set_experiment(experiment_name)
    
    print(f"[INFO] Tracking URI: {tracking_uri}")
    print(f"[INFO] Experiment Name: {experiment_name}")

    # --- 3. Hyperparameter Tuning (GridSearch) ---
    print("[INFO] Memulai Hyperparameter Tuning...")
    
    # Definisikan model dasar
    rf = RandomForestClassifier(random_state=42)
    
    # Definisikan grid parameter yang mau dicoba
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Jalankan Grid Search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"[INFO] Tuning Selesai. Parameter Terbaik: {best_params}")

    # --- 4. Manual Logging ke MLflow ---
    # Kita mulai run baru untuk mencatat model TERBAIK saja
    with mlflow.start_run(run_name="Best_Model_Tuning"):
        
        # A. Log Hyperparameters (Manual)
        # Mencatat parameter terbaik hasil tuning
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Tambahan param standar
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # B. Evaluasi Model
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"[INFO] Metrics -> Acc: {accuracy:.4f}, F1: {f1:.4f}")

        # C. Log Metrics (Manual)
        # Mencatat hasil evaluasi agar muncul di grafik MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_weighted", f1)
        
        # D. Log Model Artifact (Manual)
        # Menyimpan file model.pkl ke MLflow
        mlflow.sklearn.log_model(best_model, "model")
        
        print("[INFO] Model terbaik dan metrics berhasil disimpan ke MLflow secara manual.")

if __name__ == "__main__":
    train_with_tuning()