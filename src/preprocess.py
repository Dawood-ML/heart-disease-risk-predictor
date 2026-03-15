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

data_params = params['data']

df = pd.read_csv("data/raw/heart_disease.csv")

print(f"Raw data shape: {df.shape}")
print(f"Missing values before cleaning:\n{df.isnull().sum()}")

# WHAT: Drop rows with missing values
# WHY: Only 6 rows have missing values (num_vessels and thal columns).
#      Imputing 2% of data adds noise. Better to drop.
df = df.dropna()
print(f"After dropping NaN : {len(df)} rows")

print("\nTarget distribution after binarization:")
print(df["target"].value_counts())
print(f"Positive rate: {df['target'].mean():.2%}")


feature_cols = [c for c in df.columns if c != 'target']
X, y = df[feature_cols].values.astype(np.float32), df['target'].values.astype(np.float32)
# ── Train / val / test split ───────────────────────────────────────────────────
# WHAT: Three-way split for proper evaluation
# WHY: Val set is used during training (early stopping, LR scheduling).
#      Test set is touched ONLY at final evaluation. Never during training.
# ALTERNATIVE: Two-way split — but then you're tuning on your test set implicitly.

seed = data_params['random_seed']
test_size = data_params['test_size']
val_size  = data_params['val_size']

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=seed,
    stratify=y
)

# val_size is fraction of remaining data
frac_temp_size = val_size / (1 - test_size)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size= frac_temp_size,
    random_state=seed,
    stratify=y_temp
)

print("\nSplit sizes:")
print(f"  Train: {len(X_train)} ({len(X_train)/len(X):.0%})")
print(f"  Val:   {len(X_val)}   ({len(X_val)/len(X):.0%})")
print(f"  Test:  {len(X_test)}  ({len(X_test)/len(X):.0%})")

# Normalize Features
scaler  =  StandardScaler()
X_train =  scaler.fit_transform(X_train)
X_val   =  scaler.transform(X_val)
X_test  =  scaler.transform(X_test)

Path('data/processed').mkdir(parents=True, exist_ok=True)

np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/X_val.npy', X_val)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_val.npy', y_val)
np.save('data/processed/y_test.npy', y_test)

# Save the scaler so the inference uses the exact same normalization
# Without this, your serving API would need to recompute mean/std from somewhere
#       This scaler is part of the model artifact
Path("models").mkdir(exist_ok=True)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names for interpretability later
with open('data/processed/feature_names.json', 'w') as f:
    json.dump(feature_cols, f)

# Save dataset stats for dvc tracking
stats  = {
    "n_train": int(len(X_train)),
    "n_val" : int(len(X_val)),
    "n_test"  : int(len(X_test)),
    "n_features": int(len(feature_cols)),
    "positive_rate_train" : float(np.mean(y_train)),
    "positive_rate_test" :  float(y_test.mean()),
    "feature_means"   : scaler.mean_.tolist(),
    "feature_stds"   : scaler.scale_.tolist(), 
}
Path('metrics')
print(stats.get("positive_rate_train"))