import class_DataProcessor, class_SeriesAnalyser, class_Trader
import numpy as np
np.random.seed(1) # NumPy
import random
import pickle
random.seed(3) # Python
print("Starting session: ")
import tensorflow as tf
tf.random.set_seed(2) # Tensorflow
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.manifold import TSNE

from keras.layers import Dense, Flatten, LSTM, Dropout


def no_clustering():
    no_cluster_series = pd.Series(0,index = df_prices_train.columns)
    pairs_all_a_all, unique_tickers = series_analyser.get_candidate_pairs(clustered_series=no_cluster_series,
                                                                pricing_df_train=df_prices_train,
                                                                pricing_df_test=df_prices_test,
                                                                min_half_life=min_half_life,
                                                                max_half_life=max_half_life,
                                                                min_zero_crosings=12,
                                                                p_value_threshold=0.01,
                                                                hurst_threshold=0.5,
                                                                subsample=subsample,
                                                                )
    print(pairs_all_a_all)

def clustering_unsupervised_learning(df_prices_train, eps, min_samples):
    df_returns = data_processor.get_return_series(df_prices_train)
    df_returns.head()
    N_PRIN_COMPONENTS = 5
    df_returns_cleaned = df_returns.dropna()

    X, explained_variance = series_analyser.apply_PCA(N_PRIN_COMPONENTS, df_returns_cleaned, 
                                                    random_state=0)#12)

    clustered_series_all, clustered_series, counts, clf = series_analyser.apply_DBSCAN(eps,
                                                                                   min_samples,
                                                                                   X,
                                                                                   df_returns)
    print("DBSCAN for eps / min_samples: ", eps, min_samples)
    plot_TSNE(X,clf, clustered_series_all)
    
    #clustered_series_all, clustered_series, counts, clf = series_analyser.apply_OPTICS(X, df_returns, min_samples=2,
    #                                                                               max_eps=eps, 
    #                                                                               cluster_method='xi')
    #print("OPTICS for eps / min_samples: ", eps, min_samples)
    #plot_TSNE(X,clf, clustered_series_all)

    cluster_size(counts)

    for clust in range(len(counts)):
        symbols = list(clustered_series[clustered_series==clust].index)
        means = np.log(df_prices_train[symbols].mean())
        series = np.log(df_prices_train[symbols]).sub(means)
        series.plot(figsize=(10,5))#title='ETFs Time Series for Cluster %d' % (clust+1))
        #plt.ylabel('Normalized log prices', size=12)
        #plt.xlabel('Date', size=12)
        plt.savefig('./images/cluster_{}.png'.format(str(clust+1)), bbox_inches='tight', pad_inches=0.1)
    
    print("Evaluating pairs:")
    pairs_unsupervised, unique_tickers = series_analyser.get_candidate_pairs(clustered_series=clustered_series,
                                                            pricing_df_train=df_prices_train,
                                                            pricing_df_test=df_prices_test,
                                                            min_half_life=min_half_life,
                                                            max_half_life=max_half_life,
                                                            min_zero_crosings=12,
                                                            p_value_threshold=0.10,
                                                            hurst_threshold=0.5,
                                                            subsample=subsample
                                                            )
    print(pairs_unsupervised)
    with open('pairs_unsupervised_learning_optical_'+file_extension+'.pkl', 'wb') as file:
    # Use pickle.dump() to serialize the list and write it to the file
        pickle.dump(pairs_unsupervised, file)

def plot_TSNE(X, clf, clustered_series_all):
    """
    This function makes use of t-SNE to visualize clusters in 2D, excluding outliers and limiting the range to [-300, 300].
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np

    # Apply t-SNE transformation
    X_tsne = TSNE(learning_rate=1000, perplexity=25, random_state=1337).fit_transform(X)

    # Filter points to be within the range [-300, 300]
    valid_indices = np.logical_and.reduce([
        X_tsne[:, 0] >= -300, X_tsne[:, 0] <= 300,
        X_tsne[:, 1] >= -300, X_tsne[:, 1] <= 300
    ])
    X_tsne = X_tsne[valid_indices]
    labels = clf.labels_[valid_indices]
    clustered_series_all = clustered_series_all.iloc[valid_indices]

    # Further exclude outliers (grey dots where clustered_series_all == -1)
    clustered_indices = clustered_series_all != -1
    X_tsne = X_tsne[clustered_indices]
    labels = labels[clustered_indices]
    clustered_series_all = clustered_series_all[clustered_indices]

    # Visualization setup
    fig = plt.figure(1, facecolor='white', figsize=(15, 15), frameon=True, edgecolor='black')
    plt.clf()

    ax = fig.add_subplot(1, 1, 1, alpha=0.9)
    ax.spines['left'].set_position('center')
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_position('center')
    ax.spines['bottom'].set_alpha(0.3)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(which='major', labelsize=18)

    # Clusters (excluding outliers)
    x = X_tsne[:, 0]
    y = X_tsne[:, 1]
    tickers = list(clustered_series_all.index)
    plt.scatter(x, y, s=200, alpha=0.75, c=labels, cmap=cm.Paired)
    for i, ticker in enumerate(tickers):
        plt.annotate(ticker, (x[i] - 20, y[i] + 12), size=15)

    # Axis and labels
    plt.xlabel('t-SNE Dim. 1', position=(0.92, 0), size=20)
    plt.ylabel('t-SNE Dim. 2', position=(0, 0.92), size=20)
    ax.set_xticks(range(-200, 201, 400))
    ax.set_yticks(range(-200, 201, 400))

    # Save and show plot
    plt.savefig('PCA_OPTICS_clustering_result_1990_2015.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()



def cluster_size(counts):
    plt.figure()
    plt.barh(counts.index+1, counts.values)
    #plt.title('Cluster Member Counts')
    plt.yticks(np.arange(1, len(counts)+1, 1))
    plt.xlabel('ETFs within cluster', size=12)
    plt.ylabel('Cluster Id', size=12);
    plt.savefig('./images/0_ETF_cluster_size.png', bbox_inches='tight', pad_inches=0.1)
    

series_analyser = class_SeriesAnalyser.SeriesAnalyser()
trader = class_Trader.Trader()
data_processor = class_DataProcessor.DataProcessor()

# intraday
#df_prices = pd.read_pickle('data/etfs/pickle/commodity_ETFs_intraday_interpolated_screened_no_outliers.pickle')
#df_prices = pd.read_pickle('data/etfs/pickle/commodity_ETFs_interpolated_screened.pickle')
df_prices = pd.read_pickle('prices_date_indexed.pkl')
subsample = 2500
min_half_life = 1 # number of points in a day
max_half_life = 252 #~number of points in a year: 78*252
file_extension = 'intraday'

df_prices_train, df_prices_test = data_processor.split_data(df_prices,
                                                            ('16-05-1996',
                                                             '31-12-2015'),
                                                            ('01-01-2016',
                                                             '31-12-2020'),
                                                            remove_nan=True)
train_val_split = '01-01-2014'

#split_index = int(len(df_prices) * 0.8)  # 80% train, 20% test
#df_prices_train = df_prices.iloc[:split_index]
#df_prices_test = df_prices.iloc[split_index:]

clustering_unsupervised_learning(df_prices_train, 0.15, 3)



