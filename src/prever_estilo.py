import os
import numpy as np
from joblib import load

# 1. Caminho da pasta src/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Caminho completo até o modelo
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "modelo_estilo_usuario.joblib")

print("Carregando modelo de:", MODEL_PATH)

modelo = load(MODEL_PATH)

def prever(tarefas_semana, taxa_conclusao, taxa_remocao, proporcao_numeradas, tempo_conclusao):
    entrada = np.array([[tarefas_semana, taxa_conclusao, taxa_remocao, proporcao_numeradas, tempo_conclusao]])
    pred = modelo.predict(entrada)[0]
    return pred

# Exemplo de teste
if __name__ == "__main__":
    estilo = prever(3, 0.90, 0.05, 0.60, 1.2)
    print("Estilo previsto:", estilo)
