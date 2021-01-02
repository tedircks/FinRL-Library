# Diable the warnings
from stable_baselines3 import A2C
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import datetime
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.environment import EnvSetup
from finrl.env.EnvMultipleStock_train import StockEnvTrain
from finrl.env.EnvMultipleStock_trade import StockEnvTrade
from finrl.model.models import DRLAgent
from finrl.trade.backtest import BackTestStats, BaselineStats, BackTestPlot, backtest_strat, baseline_strat
from finrl.trade.backtest import backtest_strat, baseline_strat
import os
import sys
import warnings
warnings.filterwarnings('ignore')

print(sys.path)
# import packages
matplotlib.use('Agg')

# df = pd.read_csv(os.path.abspath('./me/datasets/data.csv'), index_col=0)
# df.sort_values(['date', 'tic'], ignore_index=True)

# fe = FeatureEngineer(df.copy(),
#                      use_technical_indicator=True,
#                      tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
#                      use_turbulence=True,
#                      user_defined_feature=False)

# df = fe.preprocess_data()

print('Begin Loading dataset')
df = pd.read_csv(os.path.abspath('./me/datasets/data_with_techs_turb.csv'), index_col=0)
print('End Loading dataset')
df.sort_values(['date', 'tic'], ignore_index=True)

train = data_split(df, '2009-01-01', '2019-01-01')
trade = data_split(df, '2019-01-01', '2020-12-01')

stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension

env_setup = EnvSetup(stock_dim=stock_dimension,
                     state_space=state_space,
                     hmax=100,
                     initial_amount=1000000,
                     transaction_cost_pct=0.001)

# training environment
env_train = env_setup.create_env_training(data=train,
                                          env_class=StockEnvTrain)

# trading environment
env_trade, obs_trade = env_setup.create_env_trading(data=trade,
                                                    env_class=StockEnvTrade)

# agent = DRLAgent(env=env_train)

# print("==============Model Training===========")
# now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
# a2c_params_tuning = {'n_steps': 5,
#                      'ent_coef': 0.005,
#                      'learning_rate': 0.0002,
#                      'verbose': 0,
#                      'timesteps': 150000}
# model_a2c = agent.train_A2C(model_name="A2C_full_train", model_params=a2c_params_tuning)
# print("============End Model Training=========")

model_a2c = A2C.load(os.path.abspath('./me/trained_models/A2C_full_train.zip'))

data_turbulence = df[(df.date < '2019-01-01') & (df.date >= '2009-01-01')]
insample_turbulence = data_turbulence.drop_duplicates(subset=['date'])

insample_turbulence.turbulence.describe()
turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)


env_trade, obs_trade = env_setup.create_env_trading(data=trade,
                                                    env_class=StockEnvTrade,
                                                    turbulence_threshold=250)

df_account_value, df_actions = DRLAgent.DRL_prediction(model=model_a2c,
                                                       test_data=trade,
                                                       test_env=env_trade,
                                                       test_obs=obs_trade)

df_actions.to_csv('./me/results/actions.csv')

# print("==============Get Backtest Results===========")
# perf_stats_all = BackTestStats(account_value=df_account_value)
# perf_stats_all = pd.DataFrame(perf_stats_all)
# # perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')

# print("==============Compare to DJIA===========")
# # S&P 500: ^GSPC
# # Dow Jones Index: ^DJI
# # NASDAQ 100: ^NDX
# BackTestPlot(df_account_value,
#              baseline_ticker='^DJI',
#              baseline_start='2019-01-01',
#              baseline_end='2020-12-01')

# print("==============Get Baseline Stats===========")
# baesline_perf_stats = BaselineStats('^DJI',
#                                     baseline_start='2019-01-01',
#                                     baseline_end='2020-12-01')
