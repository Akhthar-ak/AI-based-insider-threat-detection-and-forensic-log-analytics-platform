import subprocess
import sys

steps = [
    "preprocessing/preprocess_logs.py",
    "preprocessing/preprocess_device.py",
    "preprocessing/feature_engineering.py",
    "detection/split_data.py",
    "detection/train_model.py",
    "detection/detect_anomalies.py",
    "forensic/timeline_builder.py",
    "forensic/database_loader.py",
]

def run_step(script_path: str):
    print(f"\nRunning: {script_path}")
    subprocess.run([sys.executable, script_path], check=True)

def main():
    for script in steps:
        run_step(script)

    print("\nPipeline completed successfully.")
    print("Run dashboard with:")
    print("streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()