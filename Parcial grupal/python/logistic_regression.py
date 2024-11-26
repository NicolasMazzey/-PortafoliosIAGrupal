import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = pd.read_csv("processed_ckd_data.csv")

# Separar características (X) y etiqueta (y)
X = data.drop(columns=["class"])
y = data["class"]

# Limpiar la columna 'y'
y = y.str.strip()

# Reemplazar etiquetas por valores numéricos
y = y.replace({"ckd": 1, "notckd": 0})
y = pd.to_numeric(y, errors="coerce")

if y.isnull().any():
    raise ValueError("La columna objetivo contiene valores no numéricos o nulos después de limpiar.")

# Convertir variables categóricas a numéricas
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Escalar características (recomendado para modelos como Logistic Regression)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Definir modelo de Regresión Logística
logreg_model = LogisticRegression(max_iter=1000, random_state=42)

# Configurar validación cruzada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
roc_auc_scores = []
all_fprs = []
all_tprs = []

# Gráficos para curvas ROC
plt.figure(figsize=(10, 7))
mean_fpr = np.linspace(0, 1, 100)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Entrenar modelo
    logreg_model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = logreg_model.predict(X_test)
    y_prob = logreg_model.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva
    
    # Guardar resultados
    accuracies.append(accuracy_score(y_test, y_pred))
    
    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    roc_auc_scores.append(roc_auc)
    
    # Interpolar tpr para el promedio de curvas ROC
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0  # Garantizar que el primer punto sea 0
    all_tprs.append(tpr_interp)
    
    # Graficar curva ROC por fold
    plt.plot(fpr, tpr, label=f"Fold {fold} (AUC = {roc_auc:.2f})")

# Calcular curva ROC promedio
mean_tpr = np.mean(all_tprs, axis=0)
mean_tpr[-1] = 1.0  # Garantizar que el último punto sea 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color="blue", linestyle="--", label=f"Mean ROC (AUC = {mean_auc:.2f})")

# Finalizar gráfico ROC
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.title("ROC Curves for Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("roc_curves_logreg.png")
plt.show()

# Matriz de confusión
all_predictions = logreg_model.predict(X)
conf_matrix = confusion_matrix(y, all_predictions)
print("Confusion Matrix:\n", conf_matrix)
ConfusionMatrixDisplay(conf_matrix).plot()

# Resultados
print("Accuracies per fold:", accuracies)
print("Mean accuracy:", np.mean(accuracies))
print("Mean AUC:", mean_auc)
