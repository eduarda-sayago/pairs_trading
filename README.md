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

After cloning the git repository, please install the required Python packages by running:

```
pip install -r requirements.txt
```

As there may be package config within your global packages, it is recommended to run python in a virtual environment.

```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Graphviz

**Graphviz** is needed for model visualization (`plot_model`).

- Download Graphviz from the official [Graphviz website](https://graphviz.gitlab.io/download/).
- After downloading and installing Graphviz, make sure to add it to your system's **PATH**.

If using **Windows**, check that the Graphviz bin directory (e.g., `C:\Program Files\Graphviz\bin`) is added to the `PATH` variable.

After installing Graphviz, install the Python interface with:

```
pip install graphviz
```
### 4. Run the Script

After setting up your environment and installing the dependencies, you can run the script:

```
python rnn_trainer.py
```

### 5. Explanation of Model Training in `rnn_trainer.py`

The model training process in `rnn_trainer.py` involves training RNN models for commodity ETF forecasting. 

1. **Training Configuration:**
   The model's configuration is set up with parameters such as the number of epochs, input features, hidden nodes, loss function, optimizer, and batch size. For example, the model is trained for 500 epochs with a batch size of 512, and a hidden layer with 50 nodes.

2. **Training the Model:**
   The script trains the RNN models using data for 5 different pairs of commodities. It splits the data into training and test sets based on specified dates and uses the configuration parameters to guide the training process.

3. **Modifying the Training:**
   To modify the training process (e.g., to change the number of epochs), simply adjust the `epochs` parameter in the model configuration. Similarly, you can modify other parameters such as the batch size, number of hidden nodes, or optimizer to fine-tune the model for different performance characteristics.

4. **Saving the Models:**
   After training, the models are saved as pickle files for later use or evaluation.

To change the training, you can update the `model_config` dictionary in the script (e.g., to increase epochs, change hidden nodes, etc.). Re-running the script will retrain the model with the updated configuration.
