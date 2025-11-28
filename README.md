Detecção de Pragas e Doenças em Folhas de Mangueira com CNNs

Este projeto implementa um sistema de visão computacional baseado em redes neurais convolucionais (CNNs) para identificar pragas e doenças em folhas de mangueira, auxiliando agricultores no diagnóstico precoce e na automação do monitoramento fitossanitário.

O projeto utiliza Python e PyTorch, explorando duas arquiteturas de CNN: AlexNet e EfficientNetB0. O dataset contém 4000 imagens distribuídas em 8 classes, organizadas em conjuntos de treino, validação e teste. Durante o treinamento, são monitoradas métricas como acurácia, precisão macro, recall, F1-score e matriz de confusão. Os modelos são treinados com batch size 32, por 10 épocas, utilizando o otimizador Adam.
