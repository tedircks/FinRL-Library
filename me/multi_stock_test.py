# Diable the warnings
from stable_baselines3 import A2C
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import datetime
import gym
from pprint import pprint
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
from stable_baselines3.common.env_checker import check_env
warnings.filterwarnings('ignore')

matplotlib.use('Agg')


def test():
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

    train_data = data_split(df, '2009-01-01', '2016-01-01')
    trade_data = data_split(df, '2017-01-01', '2020-12-01')

    # params
    stock_dimension = len(train_data.tic.unique())
    state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension

    hmax = 100
    starting_capital = 1000000
    transaction_cost_pct = 0.001
    reward_scaling = 1e-4
    technical_indicator_list = config.TECHNICAL_INDICATORS_LIST

    env_setup = EnvSetup(stock_dim=stock_dimension,
                         state_space=state_space,
                         hmax=100,
                         initial_amount=1000000,
                         transaction_cost_pct=0.001)

    # training environment
    # env_train = StockEnvTrain(df=train_data,
    #                           stock_dim=stock_dimension,
    #                           hmax=hmax,
    #                           initial_amount=starting_capital,
    #                           transaction_cost_pct=transaction_cost_pct,
    #                           reward_scaling=reward_scaling,
    #                           state_space=state_space,
    #                           action_space=stock_dimension,
    #                           tech_indicator_list=technical_indicator_list,
    #                           turbulence_threshold=250,
    #                           day=0
    #                           )
    # transition to make
    # https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
    env_train = gym.make('multi-stock-train-v0',
                         df=train_data,
                         stock_dim=stock_dimension,
                         hmax=hmax,
                         initial_amount=starting_capital,
                         transaction_cost_pct=transaction_cost_pct,
                         reward_scaling=reward_scaling,
                         state_space=state_space,
                         action_space=stock_dimension,
                         tech_indicator_list=technical_indicator_list,
                         turbulence_threshold=250,
                         day=0)

    check_env(env_train)
    """
    Value:  503.1598815917969
    Params: 
        gamma: 0.0003276859847400128
        max_grad_norm: 1.573783455298761
        gae_lambda: 0.004013533056897422
        exponent_n_steps: 4
        lr: 0.18030807662070017
        ent_coef: 1.1401509020235004e-07
        ortho_init: True
        net_arch: tiny
        activation_fn: relu
    User attrs:
        gamma_: 0.99967231401526
        gae_lambda_: 0.9959864669431026
        n_steps: 16
    """
    # --------------- Training
    agent = DRLAgent(env=env_train)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    # a2c_params_tuning = {'n_steps': 512,
    #                      'ent_coef': 0.005,
    #                      'learning_rate': 0.0002,
    #                      'verbose': 0,
    #                      'timesteps': 150000}
    a2c_params_tuning = {
        "n_steps": 128,
        "gamma": 0.9913,
        "gae_lambda": 0.9987,
        "learning_rate": 0.0008,
        "ent_coef": 0.0399,
        "max_grad_norm": 0.8688,
        'verbose': 0,
        'timesteps': 20000,
        "policy_kwargs": {
            "net_arch": 'tiny',
            "activation_fn": 'tanh',
            "ortho_init": True,
        }
    }
    model_a2c = agent.train_A2C(model_name="A2C_full_train_{}".format(now), model_params=a2c_params_tuning, save=False)
    print("============End Model Training=========")

    # model_a2c = A2C.load(os.path.abspath('./me/trained_models/A2C_full_train.zip'))

    # print(calu)
    # data_turbulence = df[(df.date < '2019-01-01') & (df.date >= '2009-01-01')]
    # insample_turbulence = data_turbulence.drop_duplicates(subset=['date'])

    # insample_turbulence.turbulence.describe()
    # turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)

    # # --------------- Trading
    env_trade, obs_trade = env_setup.create_env_trading(data=trade_data,
                                                        env_class=StockEnvTrade,
                                                        turbulence_threshold=250)

    # # --------------- Predict
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=model_a2c,
                                                           test_data=trade_data,
                                                           test_env=env_trade,
                                                           test_obs=obs_trade)

    print(df_account_value)
    # df_actions.to_csv('./me/results/actions.csv')

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


if __name__ == "__main__":
    test()
