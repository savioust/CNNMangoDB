Detecção de Pragas e Doenças em Folhas de Mangueira com CNNs

Este projeto implementa um sistema de visão computacional baseado em redes neurais convolucionais (CNNs) para identificar pragas e doenças em folhas de mangueira, auxiliando agricultores no diagnóstico precoce e na automação do monitoramento fitossanitário.

O projeto utiliza Python e PyTorch, explorando duas arquiteturas de CNN: AlexNet e EfficientNetB0. O dataset contém 4000 imagens distribuídas em 8 classes, organizadas em conjuntos de treino, validação e teste. Durante o treinamento, são monitoradas métricas como acurácia, precisão macro, recall, F1-score e matriz de confusão. Os modelos são treinados com batch size 32, por 10 épocas, utilizando o otimizador Adam.

Como usar?

1 - Clonar o repositório do GitHub:
git clone https://github.com/savioust/CNNMangoDB.git

2 - Abra o terminal do Visual Studio Code (ou outro terminal) e execute o seguinte comando para instalar as dependências:
cd CNNMangoDB
python -m pip install --upgrade pip
pip install -r requirements.txt

3 - Execute o script dataset_loader.py para carregar e organizar os dados do dataset.
cd .\scripts\
python dataset_loader.py

4 - Execute o script train.py diretamente da pasta CNNMangoDB para treinar os modelos (AlexNet e EfficientNetB0) e gerar os pesos salvos.
cd ..
python scripts\train.py

5 - Execute o script visualizer.py para gerar gráficos de evolução de loss, acurácia, F1-score e outras métricas.

Link do dataset utilizado:
https://www.kaggle.com/datasets/warcoder/mango-leaf-disease-dataset
