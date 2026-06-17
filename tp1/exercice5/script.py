import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# 2. Charger le dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Taille du dataset :", X.shape)
print("Classes :", np.unique(y))

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Standardisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("Précision K-NN :", accuracy_score(y_test, y_pred_knn))

# 6. classificateur SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("Précision SVM :", accuracy_score(y_test, y_pred_svm))

# 8. Matrice de confusion SVM
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm)
plt.title("Matrice SVM")
plt.savefig("confusion_svm.png")
plt.close()

