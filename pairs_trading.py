import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import class_SeriesAnalyser, class_Trader, class_DataProcessor

# just set the seed for the random number generator
np.random.seed(107)

series_analyser = class_SeriesAnalyser.SeriesAnalyser()
trader = class_Trader.Trader()
data_processor = class_DataProcessor.DataProcessor()

df_prices = pd.read_pickle('prices_date_indexed.pkl')

df_prices_train, df_prices_test = data_processor.split_data(df_prices,
                                                            ('16-05-1996',
                                                             '31-12-2015'),
                                                            ('01-01-2016',
                                                             '31-12-2020'),
                                                            remove_nan=True)
train_val_split = '01-01-2014'

with open('pairs_unsupervised_learning_optical_intraday.pkl',
          'rb') as handle:
    pairs = pickle.load(handle)

# intraday
n_years_val = round(len(df_prices_train[train_val_split:])/(240))
print(df_prices_train[train_val_split:])

train_results_without_costs, train_results_with_costs, performance_threshold_train = \
        trader.apply_trading_strategy(pairs, 
                                       'fixed_beta',
                                        2,#entry_multiplier,
                                        0,#exit_multiplier,
                                        test_mode=False,
                                        train_val_split=train_val_split
                                       )

sharpe_results_threshold_train_nocosts, cum_returns_threshold_train_nocosts = train_results_without_costs
sharpe_results_threshold_train_w_costs, cum_returns_threshold_train_w_costs = train_results_with_costs

ticker_segment_dict = {'UN': 'UN', 'BGG': 'BGG', 'HSY': 'HSY', 'WLB': 'WLB', 'XEL': 'XEL', 'DD': 'DD', 'T': 'T', 'NC': 'NC', 'F': 'F', 'GWW': 'GWW', 'K': 'K', 'KMB': 'KMB', 'LLY': 'LLY', 'LNC': 'LNC', 'MHFI': 'MHFI', 'NOC': 'NOC', 'EIX': 'EIX', 'SLB': 'SLB', 'YRCW': 'YRCW', 'VFC': 'VFC', 'SPXC': 'SPXC', 'TAP': 'TAP', 'GCO': 'GCO', 'CNA': 'CNA', 'BF_B': 'BF_B', 'MRO': 'MRO', 'KATE': 'KATE', 'FBNIQ': 'FBNIQ', 'USW': 'USW', '571300Q': '571300Q', 'AA': 'AA', 'AXP': 'AXP', 'VZ': 'VZ', 'BA': 'BA', 'CAT': 'CAT', 'JPM': 'JPM', 'CVX': 'CVX', 'KO': 'KO', 'DIS': 'DIS', 'XOM': 'XOM', 'GE': 'GE', 'HPQ': 'HPQ', 'HD': 'HD', 'IBM': 'IBM', 'JNJ': 'JNJ', 'MCD': 'MCD', 'MRK': 'MRK', 'MMM': 'MMM', 'BAC': 'BAC', 'PFE': 'PFE', 'PG': 'PG', 'UTX': 'UTX', 'WMT': 'WMT', 'TKR': 'TKR', 'C': 'C', 'AIG': 'AIG', 'HON': 'HON', 'MO': 'MO', 'INTC': 'INTC', 'IP': 'IP', 'ABT': 'ABT', 'APD': 'APD', 'AEP': 'AEP', 'HES': 'HES', 'ADM': 'ADM', 'CCK': 'CCK', 'AVY': 'AVY', 'AVP': 'AVP', 'BHI': 'BHI', 'BLL': 'BLL', 'BCR': 'BCR', 'BAX': 'BAX', 'BDX': 'BDX', 'BMS': 'BMS', 'HRB': 'HRB', 'BMY': 'BMY', 'RAD': 'RAD', 'CPB': 'CPB', '9876566D': '9876566D', 'CI': 'CI', 'CLX': 'CLX', 'CL': 'CL', 'CSC': 'CSC', 'CAG': 'CAG', 'ED': 'ED', 'GLW': 'GLW', 'CMI': 'CMI', 'SCI': 'SCI', 'TGT': 'TGT', 'DE': 'DE', 'D': 'D', 'DOV': 'DOV', 'DOW': 'DOW', 'DUK': 'DUK', 'ETN': 'ETN', 'ECL': 'ECL', 'PKI': 'PKI', 'EMR': 'EMR', 'ETR': 'ETR', 'FDX': 'FDX', 'FMC': 'FMC', 'NEE': 'NEE', 'TGNA': 'TGNA', 'GPS': 'GPS', 'GD': 'GD', 'GIS': 'GIS', 'GPC': 'GPC', 'GT': 'GT', 'HAL': 'HAL', 'HRS': 'HRS', 'HP': 'HP', 'CNP': 'CNP', 'HUM': 'HUM', 'ITW': 'ITW', 'IR': 'IR', 'FL': 'FL', 'IFF': 'IFF', 'JCI': 'JCI', 'KR': 'KR', 'LB': 'LB', 'LOW': 'LOW', 'MMC': 'MMC', 'MAS': 'MAS', 'MDT': 'MDT', 'CVS': 'CVS', 'MSI': 'MSI', 'THC': 'THC', 'NWL': 'NWL', 'NEM': 'NEM', 'NSC': 'NSC', 'WFC': 'WFC', 'NUE': 'NUE', 'OXY': 'OXY', 'OKE': 'OKE', 'PCG': 'PCG', 'PH': 'PH', 'JCP': 'JCP', 'PEP': 'PEP', 'EXC': 'EXC', 'NYT': 'NYT', 'PHM': 'PHM', 'PBI': 'PBI', 'PNC': 'PNC', 'PPG': 'PPG', 'PEG': 'PEG', 'RTN': 'RTN', 'RDC': 'RDC', 'R': 'R', 'LUB': 'LUB', 'SHW': 'SHW', 'SNA': 'SNA', 'SO': 'SO', 'STJ': 'STJ', 'SWK': 'SWK', 'STI': 'STI', 'SVU': 'SVU', 'SYY': 'SYY', 'CAL': 'CAL', 'TXT': 'TXT', 'TJX': 'TJX', 'TMK': 'TMK', 'TYC': 'TYC', 'UNP': 'UNP', 'WBA': 'WBA', 'BSET': 'BSET', 'WY': 'WY', 'WHR': 'WHR', 'WMB': 'WMB', 'XRX': 'XRX', 'WOR': 'WOR', 'JWN': 'JWN', 'NEU': 'NEU', 'NL': 'NL', 'HST': 'HST', 'BCO': 'BCO', 'USG': 'USG', 'TRV': 'TRV', 'VVI': 'VVI', '291784Q': '291784Q', 'NKE': 'NKE', 'DTE': 'DTE', 'TXN': 'TXN', 'ITT': 'ITT', 'MDP': 'MDP', 'HAS': 'HAS', 'KBH': 'KBH', 'MAT': 'MAT', 'RRD': 'RRD', 'ASH': 'ASH', 'UIS': 'UIS', 'DDS': 'DDS', 'ADP': 'ADP', 'BC': 'BC', 'CA': 'CA', 'NAV': 'NAV', 'LPX': 'LPX', 'CTB': 'CTB', 'AAPL': 'AAPL', 'ADSK': 'ADSK', 'CMCSA': 'CMCSA', 'ORCL': 'ORCL', 'PCAR': 'PCAR', 'CR': 'CR', 'DLX': 'DLX', 'MDR': 'MDR', 'PBY': 'PBY', 'USB': 'USB', '1958Q': '1958Q', 'AMGN': 'AMGN', 'IPG': 'IPG', 'ABX': 'ABX', 'CSCO': 'CSCO', 'MSFT': 'MSFT', 'UNM': 'UNM', 'CBS': 'CBS', 'LUV': 'LUV', 'BK': 'BK', 'L': 'L', 'COP': 'COP', 'AMAT': 'AMAT', 'CMA': 'CMA', 'PPL': 'PPL', 'AON': 'AON', 'EMC': 'EMC', 'FITB': 'FITB', 'MBI': 'MBI', 'EFX': 'EFX', 'SCHW': 'SCHW', 'TMO': 'TMO', 'ADBE': 'ADBE', 'PTC': 'PTC', 'HLS': 'HLS', 'APC': 'APC', 'PGR': 'PGR', 'HBAN': 'HBAN', 'APA': 'APA', 'SNV': 'SNV', 'KLAC': 'KLAC', 'CINF': 'CINF', 'BIG': 'BIG', 'BEN': 'BEN', 'SLM': 'SLM', 'NTRS': 'NTRS', 'CCE': 'CCE', 'DHR': 'DHR', 'PAYX': 'PAYX', 'SPLS': 'SPLS', 'AFL': 'AFL', 'BBY': 'BBY', 'KSU': 'KSU', 'CTL': 'CTL', 'CMS': 'CMS', 'VMC': 'VMC', 'LEG': 'LEG', 'PNW': 'PNW', 'TER': 'TER', 'TROW': 'TROW', 'SXCL': 'SXCL', 'ADI': 'ADI', 'XLNX': 'XLNX', 'LLTC': 'LLTC', 'MXIM': 'MXIM', 'HOG': 'HOG', 'TIF': 'TIF', 'ALTR': 'ALTR', 'EOG': 'EOG', 'HOT': 'HOT', 'NI': 'NI', 'NBR': 'NBR', 'DVN': 'DVN', 'RHI': 'RHI', 'CTAS': 'CTAS', 'FISV': 'FISV', 'ZION': 'ZION', 'TE': 'TE', 'PCL': 'PCL', 'EA': 'EA', 'SYMC': 'SYMC', 'MYL': 'MYL', 'MUR': 'MUR', 'LEN': 'LEN', 'VNO': 'VNO', 'HAR': 'HAR', 'LM': 'LM', 'CELG': 'CELG', 'STR': 'STR', 'PCP': 'PCP', 'VAR': 'VAR', 'JEC': 'JEC', 'LUK': 'LUK', 'NBL': 'NBL', 'TSO': 'TSO', 'TSS': 'TSS', 'GHC': 'GHC', 'EXPD': 'EXPD', 'RRC': 'RRC', 'COG': 'COG', 'HCP': 'HCP', 'SWN': 'SWN', 'FAST': 'FAST', 'EQT': 'EQT', 'SCG': 'SCG', 'WEC': 'WEC', 'PBCT': 'PBCT', 'XRAY': 'XRAY', 'HRL': 'HRL', 'ES': 'ES', 'ARG': 'ARG', 'CLF': 'CLF', 'ROST': 'ROST', 'CERN': 'CERN', 'GAS': 'GAS', 'LRCX': 'LRCX', '1288453D': '1288453D', 'PVH': 'PVH', 'SWKS': 'SWKS', 'JBHT': 'JBHT', '1436513D': '1436513D', 'SPGI': 'SPGI'}
results, pairs_summary = trader.summarize_results(sharpe_results_threshold_train_w_costs,
                                                  cum_returns_threshold_train_w_costs,
                                                  performance_threshold_train,
                                                  pairs, ticker_segment_dict,
                                                  n_years_val)