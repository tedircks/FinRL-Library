from gym.envs.registration import register

register(
    id='multi-stock-train-v0',
    entry_point='finrl.env:StockEnvTrain',
)

register(
    id='multi-stock-trade-v0',
    entry_point='finrl.env:StockEnvTrade',
)
