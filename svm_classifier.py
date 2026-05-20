# ============================================================
# SVM Cat vs Dog Classifier
# SkillCraft Technology - Task 03
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SETTINGS
# ============================================================
IMG_SIZE = 64       # Resize all images to 64x64
MAX_IMAGES = 1000   # Use 500 cats + 500 dogs (faster training)

# ============================================================
# STEP 1: Load Images
# Dataset folder structure:
# dataset/
#   train/
#     cats/  (cat images)
#     dogs/  (dog images)
# ============================================================

def load_images(folder, label, max_count):
    data, labels = [], []
    files = os.listdir(folder)[:max_count]
    for fname in files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(img.flatten())
            labels.append(label)
    return data, labels

print("Loading images...")
cat_data, cat_labels = load_images('dataset/train/cats', 0, MAX_IMAGES//2)
dog_data, dog_labels = load_images('dataset/train/dogs', 1, MAX_IMAGES//2)

X = np.array(cat_data + dog_data)
y = np.array(cat_labels + dog_labels)
print(f"Total images loaded: {len(X)}")

# ============================================================
# STEP 2: Split Data
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# STEP 3: Scale + PCA
# ============================================================
print("Scaling and applying PCA...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

pca = PCA(n_components=100, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
print(f"PCA: reduced to {X_train_pca.shape[1]} components")

# ============================================================
# STEP 4: Train SVM
# ============================================================
print("Training SVM model...")
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm.fit(X_train_pca, y_train)
print("Training complete!")

# ============================================================
# STEP 5: Evaluate
# ============================================================
y_pred = svm.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"RESULTS")
print(f"{'='*50}")
print(f"Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Cat', 'Dog']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Cat','Dog'])
fig, ax = plt.subplots(figsize=(6,5))
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix - SVM Cat vs Dog')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("[Saved] confusion_matrix.png")

# ============================================================
# STEP 6: Save Model
# ============================================================
import joblib
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
print("[Saved] svm_model.pkl, scaler.pkl, pca.pkl")
print("\n✅ Task 03 Complete!")