# UFRGS Pairs_Trading

All the code in this project is based on https://github.com/simaomsarmento/PairsTrading/ (13.01.2025) by Simão Nava de Moraes Sarmento.

This project requires Python 3.8 (or later) and various Python packages to be installed. Please follow the instructions below to set up the environment and run the script.

## Prerequisites

Make sure you have the following installed:

- Python 3.8 (x64) or later.
- `pip` (Python's package installer).

### 1. Install Python 3.8 (or later)
If you don't have Python 3.8 or later, download it from the official [Python website](https://www.python.org/downloads/).

- During installation, ensure the box "Add Python to PATH" is checked.

### 2. Install Required Packages

As there may be package config within your global packages, it is recommended to run python and install dependencies in a virtual environment.

```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Script

After setting up your environment and installing the dependencies, you can run the script:

1. For the clustering, run:
```
python pairs_clustering.py
```

2. To perform the forecasting, run:
```
python rnn_trainer.py
```
