Projeto: Identificação de Estilo de Usuário com Machine Learning
Este projeto utiliza um modelo de Árvore de Decisão para identificar o estilo de comportamento do usuário com base nos seus padrões de criação e manipulação de tarefas. Foi desenvolvido como parte do Projeto Integrador, envolvendo Machine Learning, arquitetura, engenharia de requisitos e gerenciamento de projetos.

1. Objetivo
Criar um modelo de Machine Learning capaz de classificar o usuário em um dos estilos:
* Organizado
* Procrastinador
* Volátil
* Minimalista
A classificação se baseia em métricas extraídas das tarefas criadas e concluídas pelos usuários.

2. Dataset
O dataset utilizado está localizado em:
data/dataset_estilos_usuarios.csv
Ele contém:
Atributo	Descrição
tarefas_semana	número de tarefas criadas por semana
taxa_conclusao	% de tarefas concluídas
taxa_remocao	% de tarefas removidas
proporcao_numeradas	% de tarefas numeradas
tempo_conclusao	tempo médio de conclusão
estilo	classe alvo (rótulo)
3. Modelo Utilizado
Modelo escolhido: DecisionTreeClassifier
Motivos:
* Alta interpretabilidade
* Gera regras claras para integração com o backend
* Ótimo para datasets pequenos
* Não requer normalização
* Fácil de visualizar
O modelo treinado é salvo em:
models/modelo_estilo_usuario.joblib

4. Estrutura do Projeto
projeto-estilo-usuario/
│
├── data/
│   └── dataset_estilos_usuarios.csv
│
├── models/
│   ├── modelo_estilo_usuario.joblib
│   └── arvore_estilo_usuario.png
│
├── src/
│   ├── treinar_modelo.py
│   ├── prever_estilo.py
│   └── visualizar_arvore.py
│
├── venv/
│
├── requirements.txt
└── README.md

 5. Instalação e Execução
1. Criar ambiente virtual
python3 -m venv venv
2. Ativar (macOS / Linux)
source venv/bin/activate
3. Instalar dependências
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

6. Treinar o Modelo
python3 src/treinar_modelo.py
Este script:
* lê o dataset
* treina o modelo
* imprime acurácia
* gera matriz de confusão
* exibe regras da árvore
* salva o modelo em /models

7. Visualizar a Árvore
python3 src/visualizar_arvore.py
Gera:
models/arvore_estilo_usuario.png

8. Fazer Predições
python3 src/prever_estilo.py
Exemplo de saída:
Estilo previsto: Organizado

9. Resultados
* Acurácia: 0.93
* Estilos reconhecidos:
    * Minimalista
    * Organizado
    * Procrastinador
    * Volátil
A árvore gerada é totalmente interpretável e adequada para explicação dos critérios de decisão.

10. Tecnologias Utilizadas
Python 3.13
pandas
scikit-learn
matplotlib
joblib
VSCode

11. Discentes
Yan Ismael Clemente Alves – 01756488
Caio Ferdnand Ribeiro Páscoa – 01610376
Francisco Vitor Santos Limas – 01726471
Lavina Helena Veras Costa – 01727471
Maria Clara Pacheco da Silva – 01515396

12. Docente
Vinícius Gonçalves

Conclusão
O projeto demonstra como o Machine Learning pode ser utilizado para identificar comportamentos e padrões de usuários a partir de métricas simples, oferecendo um modelo interpretável, eficiente e pronto para integração em aplicações reais.
