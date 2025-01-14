# UFRGS Pairs_Trading_ML_e_PCA

Todo o código deste projeto é baseado em https://github.com/simaomsarmento/PairsTrading/ (13.01.2025) por Simão Nava de Moraes Sarmento.

É possível acessar o GitHub com o presente projeto [aqui](http://github.com/eduarda-sayago/pairs_trading)

O projeto requer Python 3.8 (or posterior) e instalação de diversos pacotes do Python. Por favor siga as instruções abaixo para organizar o ambiente e rodar o script.
 
## Pré-requisitos

É necessário ter os seguintes programas instalados:

- A versão do Python 3.9 (x64).
- `pip` (instalador de pacotes do Python).

- Tensorflow. Para instalar o tensorflow siga as instruções [aqui](https://www.tensorflow.org/install/pip)

### 1. Instalar versão Python 3.9 
Como o pacote tensorflow só consegue operar em versões específicas do Python, por favor instale a versão 3.9 [aqui](https://www.python.org/downloads/release/python-390/).

- Durante a instalação, tenha certeza que a caixa "Adicione o Python ao PATH" está selecionada, do contrário o pip não vai funcionar.

### 2. Instale os pacotes necessários

Como pode haver configurações de pacotes entre seu ambiente global, é recomendado que se rode o python e instale as dependências em um ambiente virtual (venv).

Se estiver no VS Code, aperte Ctrl Shift P e selecione View: Toggle Terminal. As linhas de código abaixo serão enviadas por ali.

O tensorflow requer também um ambiente virtual limpo para isolar o código.
```
python -m venv tensorflow_env
```
```
.\tensorflow_env\Scripts\activate
```

Installe os requisitos:
```
pip install -r requirements.txt
```

O trecho instalará os pacotes necessários dentro do ambiente virtual (executar no Terminal).

### 3. Rode o Script

Depois de configurar o ambiente e instalar o que é necessário, você pode rodar o script:

1. Para o processo de clustering, rode:
```
python pairs_clustering.py
```

2. Para performar o trade, rode:
```
python pairs_trading.py
```

3. Para realizar a previsão, rode:
```
python rnn_trainer.py
```

### 4. Explicação - resultados gráficos

- Processo de clustering com valores epsulon = 0.12 (distância máx. entre pontos para serem agrupados juntos), min_samples = 3 (mínimo de pontos para ser considerado um cluster)
- Gráfico com tamanho dos clusters em stocks_per_cluster
- Gráficos dos clusters visíveis na pasta cluster_charts
- Distribuição de ações em PCA_OPTICS_clustering_result_1990_2015