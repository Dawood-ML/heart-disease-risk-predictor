import pandas as pd
import yaml
import os
from pathlib import Path

# Download the UCI dataset
# Scripted downloads are more reproducible than manual download

with open('params.yaml') as f:
    params = yaml.safe_load(f)

# get data params
data_params = params['data']
print("I got the data ")
Path('data/raw').mkdir(parents=True, exist_ok=True)

# Column names from UCI documentation
# These are the 13 features used by Cleveland clinic — the "processed" version
COLUMNS = [
    "age",              # age in years
    "sex",              # 1=male, 0=female
    "chest_pain_type",  # 1=typical angina, 2=atypical angina, 3=non-anginal, 4=asymptomatic
    "resting_bp",       # resting blood pressure (mm Hg)
    "cholesterol",      # serum cholesterol (mg/dl)
    "fasting_blood_sugar",  # >120 mg/dl: 1=true, 0=false
    "rest_ecg",         # 0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy
    "max_heart_rate",   # maximum heart rate achieved
    "exercise_angina",  # exercise induced angina: 1=yes, 0=no
    "st_depression",    # ST depression induced by exercise relative to rest
    "st_slope",         # slope of peak exercise ST segment: 1=upsloping, 2=flat, 3=downsloping
    "num_vessels",      # number of major vessels colored by fluoroscopy (0-3)
    "thal",             # 3=normal, 6=fixed defect, 7=reversible defect
    "target"            # diagnosis: 0=no disease, 1-4=disease present (we binarize)
]

print(f"Downloading dataset from UCI ML Repository...")
df = pd.read_csv(
    data_params["url"],
    names=COLUMNS,
    na_values="?"   # WHAT: UCI uses "?" for missing values
)
print(f"Downloaded {len(df)} rows, {df.shape[1]} columns")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Class distribution:\n{df['target'].value_counts().sort_index()}")

df.to_csv(data_params['raw_file'], index=False)
print(f"Saved to {data_params['raw_file']}")