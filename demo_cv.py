import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc

# Set random seed
np.random.seed(42)

# Generate data
C1_count, C2_count = 40, 10
X1 = np.random.randn(C1_count, 2)
X2 = np.vstack([2 + 0.5 * np.random.randn(C2_count // 2, 2), -2 + 0.5 * np.random.randn(C2_count - C2_count // 2, 2)])
X = np.vstack([X1, X2])
class_labels = np.hstack([np.ones(C1_count), np.full(C2_count, 2)])

# Visualization function
def plot_data(X, labels, title, subplot_idx):
    plt.subplot(1, 6, subplot_idx)
    plt.scatter(X[labels == 1, 0], X[labels == 1, 1], marker='*', color='b', label='Class 1')
    plt.scatter(X[labels == 2, 0], X[labels == 2, 1], marker='o', color='r', label='Class 2')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.grid()

# Perform Cross-Validation Methods
def perform_cv(cv, title):
    plt.figure(figsize=(15, 3))
    plot_data(X, class_labels, "All Samples", 1)
    perf_curve_outputs = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, class_labels), start=1):
        X_train, y_train = X[train_idx], class_labels[train_idx]
        X_test, y_test = X[test_idx], class_labels[test_idx]
        
        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        scores = knn.predict(X_test)
        
        # Store results
        perf_curve_outputs.append((y_test, scores))
        plot_data(X_test, y_test, f"Fold: {fold}", fold + 1)
    
    # Compute AUC
    y_true, y_scores = zip(*perf_curve_outputs)
    y_true, y_scores = np.concatenate(y_true), np.concatenate(y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=2)
    auc_value = auc(fpr, tpr)
    print(f"{title} AUC: {auc_value:.4f}")
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# Standard K-Fold
kf = KFold(n_splits=5, shuffle=False)
perform_cv(kf, "K-Fold Cross-Validation")

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
perform_cv(skf, "Stratified K-Fold Cross-Validation")

import numpy as np

import numpy as np

def dobscv(X, y, n):
    """
    Distribution-Balanced Stratified Cross-Validation (DB-SCV)
    
    Args:
        X (numpy.ndarray): Feature matrix (N samples, D features)
        y (numpy.ndarray): Class labels (N,)
        n (int): Number of folds

    Returns:
        list of dict: Each dict contains 'train_idx' and 'test_idx' arrays.
    """
    num_samples = len(y)
    solution = np.full(num_samples, -1, dtype=int)  # Stores fold assignments
    classes = np.unique(y)
    fold = 0  # Tracks the next fold assignment

    for class_label in classes:
        indices = np.where(y == class_label)[0]  # Extract indices for current class
        np.random.shuffle(indices)  # Randomize to avoid ordering bias
        
        while len(indices) > 0:
            # Compute distances to the first sample in the list
            distances = np.sum((X[indices] - X[indices[0]]) ** 2, axis=1)
            nearest_indices = np.argsort(distances)[:min(n, len(indices))]  # Select n nearest neighbors
            
            # Assign these samples to folds cyclically
            fold_indices = (fold + np.arange(len(nearest_indices))) % n  # Ensure 0-based fold indexing
            solution[indices[nearest_indices]] = fold_indices  
            fold += len(nearest_indices)
            
            # Remove assigned samples
            indices = np.delete(indices, nearest_indices)
    
    # Convert to `cvpartition`-like structure
    cv = []
    for f in range(n):
        test_idx = np.where(solution == f)[0]
        train_idx = np.where(solution != f)[0]
        
        # Ensure valid train/test split
        if len(test_idx) == 0:
            print(f"Warning: Fold {f} has no test samples. Redistributing...")
            test_idx = np.random.choice(train_idx, size=max(1, len(train_idx) // (n - 1)), replace=False)
            train_idx = np.setdiff1d(train_idx, test_idx)

        cv.append({'train_idx': train_idx, 'test_idx': test_idx})
    
    return cv


# Distribution-Balanced Stratified Cross-Validation
dbscv_folds = dobscv(X, class_labels, 5)
plt.figure(figsize=(15, 3))
plot_data(X, class_labels, "All Samples", 1)
perf_curve_outputs = []

for fold, split in enumerate(dbscv_folds, start=1):
    print(f"Fold {fold}: Train={len(split['train_idx'])}, Test={len(split['test_idx'])}")
    train_idx, test_idx = split['train_idx'], split['test_idx']
    X_train, y_train = X[train_idx], class_labels[train_idx]
    X_test, y_test = X[test_idx], class_labels[test_idx]
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    scores = knn.predict_proba(X_test)[:, 1]
    
    perf_curve_outputs.append((y_test, scores))
    plot_data(X_test, y_test, f"Fold: {fold}", fold + 1)

# Compute AUC
y_true, y_scores = zip(*perf_curve_outputs)
y_true, y_scores = np.concatenate(y_true), np.concatenate(y_scores)
fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=2)
auc_value = auc(fpr, tpr)
print(f"DBSCV AUC: {auc_value:.4f}")
plt.savefig("DBSCV.png")
plt.show()
