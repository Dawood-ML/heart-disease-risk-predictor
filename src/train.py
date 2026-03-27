import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
import json
import pickle
import mlflow
import mlflow.pytorch
from pathlib import Path
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from model import HeartDiseaseClassifier

# Load the config
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

model_cfg  = params['model']
train_cfg  = params['training']
mlflow_cfg = params['mlflow']
data_cfg   = params['data']

# Load preprocessed data
X_train, y_train, X_val, y_val = torch.tensor(np.load('data/processed/X_train.npy'), dtype=torch.float32), \
                                 torch.tensor(np.load('data/processed/y_train.npy'), dtype=torch.float32), \
                                 torch.tensor(np.load('data/processed/X_val.npy'), dtype=torch.float32),   \
                                 torch.tensor(np.load('data/processed/y_val.npy'), dtype=torch.float32)
print("I worked here, lol.")

train_dataset, val_dataset = TensorDataset(X_train, y_train), \
                             TensorDataset(X_val, y_val)

print("I work here as well")

#DataLoader wraps a dataset and handles batching, shuffling
train_loader = DataLoader(
    train_dataset,
    batch_size = train_cfg['batch_size'],
    shuffle=True,
    drop_last=True # Drop final batch if smaller than batch size
)

val_loader = DataLoader(
    val_dataset,
    batch_size = len(val_dataset),
    shuffle=False,
)

# model, loss, optimizer
model = HeartDiseaseClassifier(
    input_dim=model_cfg['input_dim'],
    hidden_dims=model_cfg['hidden_dims'],
    dropout_rate=model_cfg['dropout_rate'],
    activation=model_cfg['activation']
)

criterion = nn.BCEWithLogitsLoss()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = train_cfg['learning_rate'],
    weight_decay = train_cfg['weight_decay']
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=train_cfg['epochs'],
    eta_min=1e-6
)

# Helper functions
def compute_metrics(logits, targets):
    """
    Compute Accuracy and AUC ROC from logits and binary targets
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
    
    probs = torch.sigmoid(logits).numpy()
    preds = 