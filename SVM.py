

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import os

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# 1. Load Dataset (Breast Cancer dataset)
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Train SVM with Linear Kernel
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
acc_linear = accuracy_score(y_test, y_pred_linear)

# Save linear confusion matrix
cm_linear = confusion_matrix(y_test, y_pred_linear)
disp_linear = ConfusionMatrixDisplay(confusion_matrix=cm_linear, display_labels=data.target_names)
disp_linear.plot(cmap=plt.cm.Blues)
plt.title("SVM (Linear Kernel) Confusion Matrix")
plt.savefig("outputs/confusion_matrix_linear.png")
plt.close()

# 5. Train SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
acc_rbf = accuracy_score(y_test, y_pred_rbf)

# Save RBF confusion matrix
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
disp_rbf = ConfusionMatrixDisplay(confusion_matrix=cm_rbf, display_labels=data.target_names)
disp_rbf.plot(cmap=plt.cm.Blues)
plt.title("SVM (RBF Kernel) Confusion Matrix")
plt.savefig("outputs/confusion_matrix_rbf.png")
plt.close()

# 6. Hyperparameter tuning (C and gamma for RBF)
C_values = [0.1, 1, 10]
gamma_values = [0.01, 0.1, 1]
results = []

for C in C_values:
    for gamma in gamma_values:
        svm_temp = SVC(kernel='rbf', C=C, gamma=gamma)
        scores = cross_val_score(svm_temp, X_scaled, y, cv=5)
        results.append((C, gamma, scores.mean()))

# Save tuning results
results_df = pd.DataFrame(results, columns=["C", "Gamma", "CV Accuracy"])
results_df.to_csv("outputs/hyperparameter_tuning.csv", index=False)

# 7. Decision Boundary Visualization (on synthetic 2D dataset)
X_vis, y_vis = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                   n_clusters_per_class=1, n_samples=200, random_state=42)
scaler_vis = StandardScaler()
X_vis_scaled = scaler_vis.fit_transform(X_vis)

# Train RBF SVM for visualization
svm_vis = SVC(kernel='rbf', C=1, gamma=0.5)
svm_vis.fit(X_vis_scaled, y_vis)

# Create mesh grid
x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_vis_scaled[:, 0], X_vis_scaled[:, 1], c=y_vis, s=20, edgecolor='k', cmap=plt.cm.coolwarm)
plt.title("SVM (RBF Kernel) Decision Boundary - Synthetic 2D Data")
plt.savefig("outputs/decision_boundary_rbf.png")
plt.close()

# 8. Save summary results
with open("outputs/results.txt", "w") as f:
    f.write(f"Linear Kernel Accuracy: {acc_linear:.4f}\n")
    f.write(f"RBF Kernel Accuracy: {acc_rbf:.4f}\n\n")
    f.write("Hyperparameter Tuning Results (C, Gamma, CV Accuracy):\n")
    f.write(results_df.to_string(index=False))

print("All outputs saved in 'outputs/' folder")
