# -*- encoding:utf-8 -*-
from __future__ import print_function

import numpy as np
import seaborn as sns
import warnings

# noinspection PyUnresolvedReferences
import abu_local_env
import abupy
import matplotlib.pyplot as plt
from abupy import ABuSymbolPd
from abupy import EMarketSourceType
from abupy import EMarketDataFetchMode
from abupy import AbuFactorBuyBreak
from abupy import AbuFactorAtrNStop
from abupy import AbuFactorPreAtrNStop
from abupy import AbuFactorCloseAtrNStop
from abupy import AbuMetricsBase
from abupy import abu
from abupy.CoreBu import ABuEnv
from abupy.IndicatorBu.ABuNDBoll import calc_boll
from abupy.IndicatorBu.ABuNDMacd import calc_macd
from abupy.IndicatorBu.ABuNDRsi import calc_rsi
from abupy.TLineBu.ABuTLWave import calc_wave_std, calc_wave_abs

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})

# 设置选股因子，None为不使用选股因子
stock_pickers = None
# 买入因子依然延用向上突破因子
buy_factors = [{'xd': 25, 'class': AbuFactorBuyBreak},
               {'xd': 42, 'class': AbuFactorBuyBreak}]
# 卖出因子继续使用上一章使用的因子
sell_factors = [
    {'stop_loss_n': 1.0, 'stop_win_n': 5.0,
     'class': AbuFactorAtrNStop},
    {'class': AbuFactorPreAtrNStop, 'pre_atr_n': 1.5},
    {'class': AbuFactorCloseAtrNStop, 'close_atr_n': 1.5}
]

abupy.env.disable_example_env_ipython()
# ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL
ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET
ABuEnv.g_market_target = abupy.EMarketTargetType.E_MARKET_TARGET_CN
ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx
ABuEnv.g_data_cache_type = abupy.EDataCacheType.E_DATA_CACHE_CSV
# ABuEnv.g_data_cache_type = abupy.EDataCacheType.E_DATA_CACHE_HDF5


def sample_a21():
    # 局部使用enable_example_env_ipython，示例
    # abupy.env.enable_example_env_ipython()
    # abupy.env.disable_example_env_ipython()
    # 如果本地有相应股票的缓存，可以使用如下代码强制使用本地缓存数据


    # 设置初始资金数
    read_cash = 1000000

    # *****************************************************************************************************************
    # 切换数据源

    # 强制走网络数据源
    # abupy.env.g_data_cache_type = EMarketDataFetchMode.E_DATA_CACHE_HDF5
    # 择时股票池
    # choice_symbols = ['601398', '600028', '601857', '601318', '600036', '000002', '600050', '600030']
    choice_symbols = ['600036']

    print(ABuSymbolPd.make_kl_df('601398', parallel=False).tail())


'''绘制趋势线'''


def sample_12():
    from abupy import ABuRegUtil

    # kl_pd = ABuSymbolPd.make_kl_df('601398', start='2015-03-01', end='2015-04-01', parallel=False, n_folds=1)
    kl_pd = ABuSymbolPd.make_kl_df('601398', start='2011-03-01', parallel=False, n_folds=1)
    deg = ABuRegUtil.calc_regress_deg(kl_pd.close.values, show=False)
    print('趋势角度:' + str(deg))
    pd = calc_wave_std(kl_pd, show=False)

    print(pd)
    print(pd.high)
    print(pd.low)
    print(pd.close)


    # print(calc_wave_std(abupy.FuSplitDataFrame.past_today_kl(kl_pd, '2016-03-01', 252), show=False))
    # plt.show()



'''绘制布林线'''


def sample_13():
    kl_pd = ABuSymbolPd.make_kl_df('601398', data_mode=ABuEnv.EMarketDataSplitMode.E_DATA_SPLIT_UNDO,
                                   start='2012-04-20', end='2018-04-20',
                                   parallel=False)
    print(kl_pd)
    # kl_pd.to_csv("/Users/juchen/abu/601398.csv")

    upper, middle, lower = calc_boll(kl_pd.close, 20, 2)
    print(middle)
    print(lower)
    rsi = calc_rsi(kl_pd.close)
    print(rsi)
    abupy.nd.boll.plot_boll_from_klpd(kl_pd)
    abupy.nd.rsi.plot_rsi_from_klpd(kl_pd)


def sample_14():
    kl_pd = ABuSymbolPd.make_kl_df('601398', parallel=False, n_folds=4)

    print(kl_pd.head())


def sample_15():
    # kl_pd = ABuSymbolPd.make_kl_df('601398', start='2017-12-01', end='2018-03-02', parallel=False, n_folds=4)
    kl_pd = ABuSymbolPd.make_kl_df('601398', parallel=False, n_folds=4)
    # print(calc_wave_abs(kl_pd, show=False))
    print(calc_wave_std(kl_pd))


def sample_macd():
    # kl_pd = ABuSymbolPd.make_kl_df('601398', start='2017-12-01', end='2018-03-02', parallel=False, n_folds=4)
    kl_pd = ABuSymbolPd.make_kl_df('601398', parallel=False, n_folds=4)
    # print(calc_wave_abs(kl_pd, show=False))
    dif, dea, bar = calc_macd(kl_pd.close)

    print(dif)
    print(dea)
    print(bar)

    for row in kl_pd.iterrows():  # 获取每行的index、row
        # print(type(row[1]))
        # print(row[1])
        # print(len(row))
        if int(row[1].key) == 0:
            continue
        pre_i = int(row[1].key) - 1
        now_i = int(row[1].key)
        if dif[pre_i] < dea[pre_i] and dif[now_i] >= dea[now_i]:
            print(row[0])

    # from abupy import nd
    # nd.macd.plot_macd_from_klpd(kl_pd)

if __name__ == "__main__":
    # sample_a21()
    sample_13()
    # sample_macd()
