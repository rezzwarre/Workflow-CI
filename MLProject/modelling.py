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
