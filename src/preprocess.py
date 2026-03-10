import pandas as pd 
import numpy as np
import yaml
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

with open('params.yaml') as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/raw/heart_disease.csv")

print(f"Raw data shape: {df.shape}")
print(f"Missing values before cleaning:\n{df.isnull().sum()}")

df = df.dropna()
