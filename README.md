# AI-Based Insider Threat Detection and Forensic Log Analytics Platform

## Overview

This project is a machine learning-based cybersecurity platform developed to detect potential insider threats by analyzing enterprise security logs. It processes user activity logs, extracts behavioral features, identifies anomalous users using unsupervised learning techniques, and presents the results through an interactive dashboard with forensic timeline analysis.

The project demonstrates the application of machine learning in User Behavior Analytics (UBA) to support security analysts in identifying suspicious activities within an organization.

---

## Features

- Enterprise log preprocessing and cleaning
- Behavioral feature engineering
- Insider threat detection using:
  - Isolation Forest
  - One-Class SVM
- Comparative analysis of anomaly detection models
- Forensic timeline reconstruction
- Interactive Streamlit dashboard
- SQLite database integration
- Threat simulation module for testing user behavior

---

## Project Structure

```
AI-Based-Insider-Threat-Detection/
│
├── dashboard/          # Streamlit dashboard
├── preprocessing/      # Data preprocessing scripts
├── detection/          # Anomaly detection models
├── forensic/           # Timeline generation and database loader
├── database/           # SQLite utilities
├── models/             # Trained ML models
├── data/
│   ├── raw/
│   └── processed/
├── app.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- SQLite
- Plotly
- Joblib

---

## Machine Learning Models

### Isolation Forest
Detects anomalous users by isolating observations through random partitioning.

### One-Class SVM
Learns the boundary of normal user behavior and identifies users outside the learned boundary as anomalies.

The outputs of both models are compared to improve confidence in anomaly detection.

---

## Workflow

1. Collect enterprise log datasets
2. Preprocess raw log files
3. Generate user behavioral features
4. Train anomaly detection models
5. Detect suspicious users
6. Generate forensic timelines
7. Store results in SQLite
8. Visualize insights using Streamlit

---

## Dataset

This project is designed to work with enterprise log datasets such as the CERT Insider Threat Dataset.

The repository **does not include the datasets** because of their large size.

Expected datasets include:

- `logon.csv`
- `file.csv`
- `device.csv`

Place these files inside:

```
data/raw/
```

before running the preprocessing pipeline.

---

## Files Excluded from GitHub

The following files and folders are intentionally excluded from the repository:

- Raw datasets (`data/raw/`)
- Processed datasets (`data/processed/`)
- SQLite databases (`*.db`)
- Trained machine learning models (`*.pkl`)
- Python virtual environments
- Cache files
- Log files

These files are generated locally after running the project.

---

## Installation

Clone the repository

```bash
git clone https://github.com/your-username/AI-Based-Insider-Threat-Detection.git
```

Move into the project directory

```bash
cd AI-Based-Insider-Threat-Detection
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the preprocessing pipeline

```bash
python preprocessing/preprocess_logs.py
python preprocessing/preprocess_device.py
python feature_engineering.py
```

Run anomaly detection

```bash
python detection/detect_anomalies.py
```

Generate forensic timeline and database

```bash
python forensic/database_loader.py
```

Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Future Improvements

- Real-time log ingestion
- SIEM integration
- Additional anomaly detection techniques
- Explainable AI (XAI) for anomaly interpretation
- Alert and notification system
- Role-based authentication

---

## Author

**Salim Akhthar**

Master of Computer Applications (Cybersecurity)

---

## License

This project is intended for educational and research purposes.
