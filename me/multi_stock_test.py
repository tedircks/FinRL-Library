# Diable the warnings
from stable_baselines3 import A2C
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import datetime
import gym
import seaborn as sns
from me.callbacks.save_best_model_cb import SaveOnBestTrainingRewardCallback
from pprint import pprint
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import BackTestStats, BaselineStats, BackTestPlot, backtest_strat, baseline_strat
from finrl.trade.backtest import backtest_strat, baseline_strat
import os
import sys
import warnings
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
warnings.filterwarnings('ignore')

matplotlib.use('TkAgg')


def test(training_data, trading_data):

    # params
    stock_dimension = len(training_data.tic.unique())
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

    # pre-make training environment
    # env_train = StockEnvTrain(params)

    # transition to make
    # https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
    env_train = gym.make('multi-stock-train-v0',
                         df=training_data,
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

    # --------------- Training

    log_dir = "me/tmp/"
    os.makedirs(log_dir, exist_ok=True)
    env_train = Monitor(env_train, log_dir)

    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)

    agent = DRLAgent(env=env_train)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    # a2c_params_tuning = {'n_steps': 512,
    #                      'ent_coef': 0.005,
    #                      'learning_rate': 0.0002,
    #                      'verbose': 0,
    #                      'timesteps': 150000}
    a2c_params_tuning = {
        "n_steps": 32,
        "gamma": 0.999304473794672,
        "gae_lambda": 0.994452346235796,
        "learning_rate": 0.00010054610987642753,
        "ent_coef": 0.00215496380633495,
        "max_grad_norm": 2.217146296318495,
        'verbose': 0,
        'timesteps': 2e5,  # 2e5
        "policy_kwargs": {
            "net_arch": 'tiny',
            "activation_fn": 'tanh',
            "ortho_init": False,
        }
    }
    model_a2c = agent.train_A2C(
        model_name="A2C_full_train_tuned{}".format(now),
        model_params=a2c_params_tuning,
        save=True,
        callback=callback)
    print("============End Model Training=========")

    # model_a2c = A2C.load(os.path.abspath('./me/tmp/best_model.zip'))

    account_value, actions = get_trade_results(env_setup, model_a2c)


def new_test():

    processed = pd.read_csv(os.path.abspath('./me/datasets/new_data_with_techs_turb.csv'), index_col=0)

    train = data_split(processed, '2009-01-01', '2018-01-01')
    trade = data_split(processed, '2018-01-01', '2021-01-01')

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    log_dir = "me/tmp/"
    os.makedirs(log_dir, exist_ok=True)

    env_train.envs[0] = Monitor(env_train.envs[0], log_dir)

    agent = DRLAgent(env=env_train)
    model_a2c = agent.get_model("a2c", verbose=0)

    trained_a2c = agent.train_model(model=model_a2c,
                                    tb_log_name='a2c',
                                    total_timesteps=100000)

    data_turbulence = processed[(processed.date < '2018-01-01') & (processed.date >= '2009-01-01')]
    insample_turbulence = data_turbulence.drop_duplicates(subset=['date'])
    turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)

    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=380, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    print("BEGIN PREDICTION")
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_a2c,
                                                           test_data=trade,
                                                           test_env=env_trade,
                                                           test_obs=obs_trade)

    print(df_account_value)

    print("END PREDICTION")


def get_trade_results(env_setup, model):
    # # --------------- Trading
    env_trade, obs_trade = env_setup.create_env_trading(data=trading_data,
                                                        env_class=StockEnvTrade,
                                                        turbulence_threshold=230)

    # # --------------- Predict
    df_account_value, df_actions = DRLAgent.DRL_prediction(model=model,
                                                           test_data=trading_data,
                                                           test_env=env_trade,
                                                           test_obs=obs_trade)

    return df_account_value, df_actions


def get_yahoo_data(start, end):
    df = YahooDownloader(start_date=start,
                         end_date=end,
                         ticker_list=config.DOW_30_TICKER).fetch_data()

    df.sort_values(['date', 'tic'], ignore_index=True)

    x = df.tic.unique()
    templ = []

    # get intersection data, smallest data
    for name, group in df.groupby('date'):
        g = group.tic.unique()
        if len(templ) == 0:
            templ = [i for i in g if i in x]
        else:
            templ = [i for i in g if i in templ]

    data_merge = pd.DataFrame(columns=list(df.columns))
    x = np.array(templ).reshape(-1, 1)
    temp_df = pd.DataFrame.from_records(x, columns=['tic'])

    for name, group in df.groupby('date'):
        temp_df['date'] = name

        result_outer = pd.merge(group, temp_df,  on=['date', 'tic'])
        result_outer = result_outer.sort_values(['date', 'tic'], ignore_index=True)

        assert len(result_outer) == len(temp_df.tic.unique())
        data_merge = data_merge.append(result_outer)

    df = data_merge

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        user_defined_feature=False)

    processed = fe.preprocess_data(df)
    processed.sort_values(['date', 'tic'], ignore_index=True)

    return processed


def get_feature_engineered_df(df):
    fe = FeatureEngineer(df.copy(),
                         use_technical_indicator=True,
                         tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
                         use_turbulence=True,
                         user_defined_feature=False)

    df = fe.preprocess_data()
    return df


def get_turbulence_threshold(df):
    data_turbulence = df[(df.date < '2019-01-01') & (df.date >= '2009-01-01')]
    insample_turbulence = data_turbulence.drop_duplicates(subset=['date'])

    insample_turbulence.turbulence.describe()
    turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)

    return turbulence_threshold


def back_test(account_value):
    print("==============Get Backtest Results===========")
    perf_stats_all = BackTestStats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    # perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')

    print("==============Compare to DJIA===========")
    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    BackTestPlot(df_account_value,
                 baseline_ticker='^DJI',
                 baseline_start='2019-01-01',
                 baseline_end='2020-12-01')

    print("==============Get Baseline Stats===========")
    baesline_perf_stats = BaselineStats('^DJI',
                                        baseline_start='2019-01-01',
                                        baseline_end='2020-12-01')


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")

    # built int
    results_plotter.plot_results([log_folder], 3e5, results_plotter.X_TIMESTEPS, "TD3 LunarLander")

    plt.show()


if __name__ == "__main__":
    # df = pd.read_csv(os.path.abspath('./me/datasets/data_with_techs_turb.csv'), index_col=0)
    # df.sort_values(['date', 'tic'], ignore_index=True)

    # training_data = data_split(df, '2009-01-01', '2017-01-01')
    # trading_data = data_split(df, '2017-01-01', '2020-12-01')

    # test(training_data, trading_data)

    # # plot_results("me/tmp/")
    # plot_results("me/tmp")

    new_test()
