# kfold_evaluate_model.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as _DepthwiseConv2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# —— CONFIGURATION —— 
DATA_DIR         = "Data"                          # Folder with one subfolder per gesture
MODEL_FILE       = r"E:\current project\AI_PROJECT\AI_MODEL\keras_model_custom2.h5"         # Your saved model
LABELS_FILE      = r"E:\current project\AI_PROJECT\AI_MODEL\labels.txt"                   # labels.txt file
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 32
N_SPLITS         = 5                               # Number of folds

# —— CUSTOM DEPTHWISE LAYER FOR LOADING SAVED MODEL —— 
class DepthwiseConv2D(_DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)

# —— LOAD YOUR SAVED MODEL ONCE —— 
if not os.path.isfile(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
model = load_model(MODEL_FILE, custom_objects={"DepthwiseConv2D": DepthwiseConv2D})

# —— READ LABELS & PREPARE DATAFRAME —— 
labels = []
with open(LABELS_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            labels.append(" ".join(parts[1:]))

# Collect file paths and true labels
filepaths, true_labels = [], []
for lbl in labels:
    class_dir = os.path.join(DATA_DIR, lbl)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        path = os.path.join(class_dir, fname)
        if os.path.isfile(path):
            filepaths.append(path)
            true_labels.append(lbl)

# Encode labels to integers
le = LabelEncoder()
y = le.fit_transform(true_labels)
X = np.array(filepaths)

# —— SET UP STRATIFIED K-FOLD —— 
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_accuracies = []

print("Evaluating saved model with Stratified K-Fold:")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    val_paths = X[val_idx]
    val_labels = y[val_idx]
    
    correct = 0
    for path, true_idx in zip(val_paths, val_labels):
        img = load_img(path, target_size=IMG_SIZE)
        x   = img_to_array(img).astype("float32") / 255.0
        x   = np.expand_dims(x, 0)
        preds = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        if pred_idx == true_idx:
            correct += 1
    
    acc = correct / len(val_paths)
    fold_accuracies.append(acc)
    print(f" Fold {fold}: {acc:.2%} accuracy on {len(val_paths)} samples")

# —— SUMMARY —— 
avg_acc = np.mean(fold_accuracies)
print(f"\nAverage accuracy across {N_SPLITS} folds: {avg_acc:.2%}")

# —— PLOT —— 
plt.figure()
plt.plot(range(1, N_SPLITS+1), fold_accuracies, marker='o')
plt.title("Saved Model Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()
