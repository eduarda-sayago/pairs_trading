# UFRGS Pairs_Trading_ML_e_PCA

Todo o código deste projeto é baseado em https://github.com/simaomsarmento/PairsTrading/ (13.01.2025) por Simão Nava de Moraes Sarmento.

O projeto requer Python 3.8 (or posterior) e instalação de diversos pacotes do Python. Por favor siga as instruções abaixo para organizar o ambiente e rodar o script.

## Pré-requisitos

É necessário ter os seguintes programas instalados:

- Python 3.8 (x64) ou posterior.
- `pip` (instalador de pacotes do Python).

### 1. Instalar versão Python 3.8 (ou posterior)
Se não houver instalado, instale em [Python website](https://www.python.org/downloads/).

- Durante a instalação, tenha certeza que a caixa "Adicione o Python ao PATH" está selecionada.

### 2. Instale os pacotes necessários

Como pode haver configurações de pacotes entre seu ambiente global, é recomendado que se rode o python e instale as dependências em um ambiente virtual (venv).

Se estiver no VS Code, aperte Ctrl Shift P e selecione View: Toggle Terminal. Então digite:

```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
O trecho instalará os pacotes necessários dentro do ambiente virtual.

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
