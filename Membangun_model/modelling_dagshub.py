import pandas as pd
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import dagshub

def train_with_dagshub():
    # --- 1. Konfigurasi DagsHub (ISI INI) ---
    # Ganti dengan Username, Repo Name, dan Token Anda
    DAGSHUB_USERNAME = "FaizFP"
    DAGSHUB_REPO_NAME = "my-first-repo"
    DAGSHUB_TOKEN = "''''" # Bisa didapat di Settings > Access Tokens
    
    # Setup Auth Environment Variables secara otomatis
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    
    # Setup Tracking URI
    # Format: https://dagshub.com/<username>/<repo>.mlflow
    remote_server_uri = f"https://dagshub.com/FaizFP/my-first-repo.mlflow"
    mlflow.set_tracking_uri(remote_server_uri)
    
    # Nama Eksperimen
    mlflow.set_experiment("Eksperimen_Online_FaizFirdausPriyanto")
    
    print(f"[INFO] Tracking URI: {remote_server_uri}")

    # --- 2. Load Data ---
    # Gunakan path absolute ke data hasil preprocessing
    input_path = r'C:\Users\faiz\Documents\tgstgs\SEMESTER 5\Dicoding\Eksperimen_SML_FaizFirdausPriyanto\preprocessing\wine_quality_clean.csv'
    
    if not os.path.exists(input_path):
        print(f"[ERROR] File tidak ditemukan di: {input_path}")
        return

    print(f"[INFO] Memuat data dari: {input_path}")
    df = pd.read_csv(input_path)

    target_col = 'quality'
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 3. Training & Manual Logging ---
    print("[INFO] Memulai Training...")
    
    with mlflow.start_run(run_name="Manual_Logging_Artifacts"):
        
        # A. Init & Train Model
        n_estimators = 100
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # B. Prediksi
        y_pred = model.predict(X_test)
        
        # C. Hitung Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"[INFO] Metrics: Acc={acc:.4f}, F1={f1:.4f}")

        # --- D. LOGGING MANUAL (Syarat: Metrics + Artifacts) ---
        
        # 1. Log Parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("algorithm", "RandomForest")
        
        # 2. Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # 3. Log Model
        mlflow.sklearn.log_model(model, "model")
        
        # --- E. ARTEFAK TAMBAHAN (Minimal 2) ---
        
        # Artefak 1: Plot Confusion Matrix
        print("[INFO] Membuat Confusion Matrix Plot...")
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Simpan gambar sementara lalu log ke MLflow
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path) # Upload ke DagsHub
        plt.close()
        
        # Artefak 2: Plot Feature Importance
        print("[INFO] Membuat Feature Importance Plot...")
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        feat_names = X.columns
        indices = importances.argsort()[::-1]
        
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), feat_names[indices], rotation=45)
        plt.title("Feature Importance")
        plt.tight_layout()
        
        fi_path = "feature_importance.png"
        plt.savefig(fi_path)
        mlflow.log_artifact(fi_path) # Upload ke DagsHub
        plt.close()
        
        # Artefak 3 (Bonus): Text Report
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact(report_path)
        
        # Bersihkan file sementara
        if os.path.exists(cm_path): os.remove(cm_path)
        if os.path.exists(fi_path): os.remove(fi_path)
        if os.path.exists(report_path): os.remove(report_path)
        
        print("[SUCCESS] Semua metrics dan artefak berhasil di-upload ke DagsHub!")

if __name__ == "__main__":
    train_with_dagshub()