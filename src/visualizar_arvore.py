import os
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from joblib import load

# 1. Caminho da pasta src/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Caminho completo até o modelo treinado
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "modelo_estilo_usuario.joblib")

print("Carregando modelo de:", MODEL_PATH)

# 3. Carregar o modelo
modelo = load(MODEL_PATH)

# 4. Plotar a árvore
plt.figure(figsize=(16, 10))
plot_tree(
    modelo,
    feature_names=[
        "tarefas_semana",
        "taxa_conclusao",
        "taxa_remocao",
        "proporcao_numeradas",
        "tempo_conclusao"
    ],
    class_names=modelo.classes_,
    filled=True,
    rounded=True
)

# 5. Salvar imagem na pasta models/
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "models", "arvore_estilo_usuario.png")
plt.tight_layout()
plt.savefig(OUTPUT_PATH)

print("Imagem salva em:", OUTPUT_PATH)
