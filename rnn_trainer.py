import class_ForecastingTrader, class_DataProcessor
import numpy as np
np.random.seed(1) # NumPy
import random
random.seed(3) # Python
import tensorflow as tf
tf.random.set_seed(2) # Tensorflow
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K


import pandas as pd
import pickle
import gc

forecasting_trader = class_ForecastingTrader.ForecastingTrader()
data_processor = class_DataProcessor.DataProcessor()

################################# READ PRICES AND PAIRS #################################
# read prices
df_prices = pd.read_pickle('cleaned_file_indexed.pkl')
# split data in training and test
split_index = int(len(df_prices) * 0.8)  # 80% train, 20% test
df_prices_train = df_prices.iloc[:split_index]
df_prices_test = df_prices.iloc[split_index:]
df_prices_train, df_prices_test = data_processor.split_data(df_prices,
                                                            ('1',
                                                             '5500'),
                                                            ('5501',
                                                             '6426'),
                                                            remove_nan=True)
# load pairs
with open('pairs_unsupervised_learning_optical_intraday.pkl', 'rb') as handle:
    pairs = pickle.load(handle)
    print(handle)
n_years_train = round(len(df_prices_train) / (240))
print('Loaded {} pairs!'.format(len(pairs)))

################################# TRAIN MODELS #################################

combinations = [(24, [50])]
hidden_nodes_names = ['50_nodes']

for i, configuration in enumerate(combinations):

    model_config = {"n_in": configuration[0],
                    "n_out": 1,
                    "epochs": 3,
                    "hidden_nodes": configuration[1],
                    "loss_fct": "mse",
                    "optimizer": "rmsprop",
                    "batch_size": 512,
                    "train_val_split": '5000',
                    "test_init": '5001'
    }
    models = forecasting_trader.train_models(pairs, model_config, model_type='rnn')

    # save models for this configuration
    with open('./rnn_models/models_n_in-' + str(configuration[0]) + '_hidden_nodes-' + hidden_nodes_names[i] +
              '.pkl','wb') as f:
        pickle.dump(models, f)

gc.collect()