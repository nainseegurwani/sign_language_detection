import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as _DepthwiseConv2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ——— CONFIGURE THESE PATHS ———
DATA_DIR    = r"E:\current project\AI_PROJECT\AI_Project\Data"         # your folder of per-gesture subfolders
LABELS_FILE = r"E:\current project\AI_PROJECT\AI_MODEL\labels.txt"   # your labels.txt
MODEL_FILE  =r"E:\current project\AI_PROJECT\AI_MODEL\keras_model_custom2.h5" # your .h5 model file

# 1) Custom object for loading your model
class DepthwiseConv2D(_DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)

# 2) Load model
if not os.path.isfile(MODEL_FILE):
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
model = load_model(MODEL_FILE, custom_objects={"DepthwiseConv2D": DepthwiseConv2D})

# 3) Read your labels into a list
labels = []
with open(LABELS_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            # drop the index, keep the name (handles multi-word labels)
            labels.append(" ".join(parts[1:]))
if not labels:
    raise ValueError("No labels loaded. Check LABELS_FILE path.")

# 4) Gather all image paths & true labels
paths, true_labels = [], []
for lbl in labels:
    cls_dir = os.path.join(DATA_DIR, lbl)
    if not os.path.isdir(cls_dir):
        continue
    for fname in os.listdir(cls_dir):
        fp = os.path.join(cls_dir, fname)
        if os.path.isfile(fp):
            paths.append(fp)
            true_labels.append(lbl)

# 5) Split 80/20 (stratified)
from sklearn.model_selection import train_test_split
p_train, p_test, l_train, l_test = train_test_split(
    paths, true_labels,
    test_size=0.2,
    stratify=true_labels,
    random_state=42
)

# 6) Run inference on test set
y_true_idx, y_pred_idx = [], []
for fp, true_lbl in zip(p_test, l_test):
    img = load_img(fp, target_size=(224,224))
    x   = img_to_array(img).astype("float32")/255.0
    x   = np.expand_dims(x, 0)
    pred = model.predict(x, verbose=0)[0]
    pi   = int(np.argmax(pred))
    y_true_idx.append(labels.index(true_lbl))
    y_pred_idx.append(pi)

# 7) Compute metrics
cm = confusion_matrix(y_true_idx, y_pred_idx)
print("\nClassification Report:\n")
print(classification_report(y_true_idx, y_pred_idx, target_names=labels))

# 8) Plot confusion matrix heatmap
plt.figure(figsize=(12,10))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            xticklabels=labels,
            yticklabels=labels,
            cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
