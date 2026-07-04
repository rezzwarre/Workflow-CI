"""
modelling.py

Melatih model klasifikasi (RandomForestClassifier) menggunakan dataset
hasil preprocessing. File ini dijalankan sebagai entry point MLflow
Project (lihat file `MLProject`) sehingga bisa dipicu otomatis oleh
GitHub Actions (Workflow CI) untuk melakukan re-training model.

Kriteria yang dipenuhi (Basic):
- Melatih model Scikit-Learn menggunakan MLflow Tracking UI lokal.
- Tidak menggunakan hyperparameter tuning.
- Menggunakan autolog dari MLflow.

Menjalankan manual:
    python modelling.py

Menjalankan sebagai MLflow Project (dari dalam folder MLProject/):
    mlflow run . --env-manager=local

--------------------------------------------------------------------
PENYESUAIAN untuk tahap Workflow-CI (dibanding versi eksperimen lokal):
1. Tracking URI "sqlite:///mlflow.db" HANYA dipasang saat script
   dijalankan secara manual (python modelling.py). Saat dijalankan
   lewat `mlflow run` (baik lokal maupun dari workflow CI), MLflow
   Project SUDAH membuat run & tracking URI-nya sendiri (env var
   MLFLOW_RUN_ID & MLFLOW_TRACKING_URI, default-nya folder ./mlruns).
   Jika kita paksa override ke sqlite di sini, terjadi bentrok
   ("Run not found") karena run_id yang sudah dibuat `mlflow run`
   tidak ada di sqlite store yang baru. Makanya sekarang dicek dulu
   apakah sedang berjalan di dalam MLflow run (lewat env MLFLOW_RUN_ID).
2. DATASET_PATH & OUTPUT_MODEL_DIR dibuat relatif terhadap lokasi
   script ini (bukan terhadap current working directory), supaya tetap
   konsisten baik dijalankan manual, lewat `mlflow run`, maupun lewat
   GitHub Actions runner yang working directory-nya bisa berbeda-beda.
3. Setelah training, model TAMBAHAN diekspor secara eksplisit ke folder
   tetap `model/` (di luar mlruns/) menggunakan `mlflow.sklearn.save_model`.
   Ini penting supaya step Docker build (`mlflow models build-docker`)
   pada workflow CI punya path model yang pasti dan tidak perlu mencari
   run_id secara dinamis.
--------------------------------------------------------------------
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "namadataset_preprocessing", "flower_morphometrics_preprocessed.csv")
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, "model")
TARGET_COLUMN = "species"


def load_preprocessed_data(path: str) -> pd.DataFrame:
    """Memuat dataset yang sudah melalui tahap preprocessing (siap latih)."""
    df = pd.read_csv(path)

    
    if "altitude_group" in df.columns and df["altitude_group"].dtype == "object":
        df["altitude_group"] = LabelEncoder().fit_transform(df["altitude_group"])

    return df


def main():
    
    running_inside_mlflow_run = "MLFLOW_RUN_ID" in os.environ

    if not running_inside_mlflow_run:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Flower_Species_Classification_Basic")

    # 2. Load dataset hasil preprocessing
    df = load_preprocessed_data(DATASET_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Aktifkan autolog MLflow untuk Scikit-Learn
    mlflow.sklearn.autolog()

    
    active_run = mlflow.start_run() if running_inside_mlflow_run else mlflow.start_run(run_name="RandomForest_Baseline_CI")

    with active_run:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

       
        test_accuracy = model.score(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy:.4f}")

        
        if os.path.exists(OUTPUT_MODEL_DIR):
            import shutil
            shutil.rmtree(OUTPUT_MODEL_DIR)
        mlflow.sklearn.save_model(model, path=OUTPUT_MODEL_DIR)
        print(f"Model juga diekspor ke: {OUTPUT_MODEL_DIR}")

    print("Training selesai. Cek hasilnya dengan menjalankan: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
