import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump

# 1. Descobrir a pasta deste arquivo (src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Montar o caminho correto até o CSV (../data/dataset_estilos_usuarios.csv)
CSV_PATH = os.path.join(BASE_DIR, "..", "data", "dataset_estilos_usuarios.csv")

print("Lendo arquivo:", CSV_PATH)

# 3. Ler o CSV (separador ; por causa do Excel em pt-BR)
df = pd.read_csv(CSV_PATH, sep=";")

# 4. Separar features (X) e rótulo (y)
X = df[[
    "tarefas_semana",
    "taxa_conclusao",
    "taxa_remocao",
    "proporcao_numeradas",
    "tempo_conclusao"
]]
y = df["estilo"]

# 5. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 6. Criar e treinar o modelo (Árvore de Decisão)
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 7. Avaliar o modelo
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

print("\nAcurácia:", acc)
print("\nClasses:", clf.classes_)
print("\nMatriz de confusão:\n", cm)

# 8. Exibir regras da árvore (para relatório / backend)
rules = export_text(clf, feature_names=X.columns.tolist())
print("\nRegras da árvore de decisão:\n")
print(rules)

# 9. Salvar o modelo treinado
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "modelo_estilo_usuario.joblib")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
dump(clf, MODEL_PATH)

print(f"\nModelo salvo em: {MODEL_PATH}")
