1 - Clonar o repositório do GitHub:
git clone https://github.com/savioust/ProjetoVisaoComputacional.git

2 - Abra o terminal do Visual Studio Code (ou outro terminal) e execute o seguinte comando para instalar as dependências:
cd ProjetoVisaoComputacional
python -m pip install --upgrade pip
pip install -r requirements.txt

3 - Execute o script dataset_loader.py para carregar e organizar os dados do dataset.
cd .\scripts\
python dataset_loader.py

4 - Execute o script train.py diretamente da pasta ProjetoVisaoComputacional para treinar os modelos (AlexNet e EfficientNetB0) e gerar os pesos salvos.
cd ..
python scripts\train.py

5 - Execute o script visualizer.py para gerar gráficos de evolução de loss, acurácia, F1-score e outras métricas.