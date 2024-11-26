import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = pd.read_csv("processed_ckd_data.csv")

# Separar características (X) y etiqueta (y)
X = data.drop(columns=["class"])
y = data["class"]

y = y.str.strip()

# Codificar etiquetas (ckd = 1, notckd = 0)
y = y.replace({"ckd": 1, "notckd": 0})

if not isinstance(y, pd.Series):
    y = pd.Series(y)

y = pd.to_numeric(y, errors="coerce")

# Validar si hay valores nulos después de la conversión
if y.isnull().any():
    raise ValueError("La columna objetivo contiene valores no numéricos o nulos.")

# StratifiedKFold con validación
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Convertir variables categóricas a numéricas
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, criterion="entropy", random_state=42)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
roc_auc_scores = []
tpr_list = []
mean_fpr = np.linspace(0, 1, 100)  # Para calcular la curva promedio
all_predictions = []
all_true_labels = []

# Gráficos para curvas ROC
plt.figure(figsize=(10, 7))
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Entrenar modelo
    rf_model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva
    
    # Guardar resultados
    accuracies.append(accuracy_score(y_test, y_pred))
    roc_auc = auc(*roc_curve(y_test, y_prob)[:2])
    roc_auc_scores.append(roc_auc)
    
    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    tpr_list.append(np.interp(mean_fpr, fpr, tpr))  # Interpolar TPR para curva promedio
    tpr_list[-1][0] = 0.0  # Asegurarse de que la curva ROC comience en (0, 0)
    
    # Graficar curva ROC por fold
    plt.plot(fpr, tpr, label=f"Fold {fold} (AUC = {roc_auc:.2f})")
    all_predictions.extend(y_pred)
    all_true_labels.extend(y_test)

# Calcular y graficar la curva ROC promedio
mean_tpr = np.mean(tpr_list, axis=0)
mean_tpr[-1] = 1.0  # Asegurarse de que la curva ROC termine en (1, 1)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color="blue", linestyle="--", label=f"Mean ROC (AUC = {mean_auc:.2f})")

# Finalizar gráfico ROC
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("roc_curves_adapted.png")
plt.show()

# Matriz de confusión
conf_matrix = confusion_matrix(all_true_labels, all_predictions)
print("Confusion Matrix:\n", conf_matrix)
ConfusionMatrixDisplay(conf_matrix).plot()

# Resultados
print("Accuracies per fold:", accuracies)
print("Mean accuracy:", np.mean(accuracies))
print("Mean AUC:", mean_auc)
